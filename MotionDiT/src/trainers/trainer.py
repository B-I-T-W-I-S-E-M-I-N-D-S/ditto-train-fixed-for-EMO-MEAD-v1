import torch
import os
import time
from tqdm import trange, tqdm
import traceback
import numpy as np

from ..utils.utils import load_json, DictAverageMeter, dump_pkl
from ..models.modules.adan import Adan
from ..models.LMDM import LMDM
from ..datasets.s2_dataset_v2 import Stage2Dataset as Stage2DatasetV2
from ..options.option import TrainOptions


class Trainer:
    def __init__(self, opt: TrainOptions):
        self.opt = opt
        self.best_val_loss = float("inf")

        print(time.asctime(), '_init_accelerate')
        self._init_accelerate()

        print(time.asctime(), '_init_LMDM')
        self.LMDM = self._init_LMDM()

        print(time.asctime(), '_init_dataset')
        self.train_loader, self.val_loader = self._init_dataloaders()

        print(time.asctime(), '_init_optim')
        self.optim = self._init_optim()

        print(time.asctime(), '_set_accelerate')
        self._set_accelerate()

        print(time.asctime(), '_init_log')
        self._init_log()

        print(time.asctime(), '_maybe_resume_training')
        self._maybe_resume_training()

    def _init_accelerate(self):
        opt = self.opt
        if opt.use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.is_main_process = self.accelerator.is_main_process
            self.process_index = self.accelerator.process_index
        else:
            self.accelerator = None
            self.device = 'cuda'
            self.is_main_process = True
            self.process_index = 0

    def _set_accelerate(self):
        if self.accelerator is None:
            return
        
        self.LMDM.use_accelerator(self.accelerator)
        self.optim = self.accelerator.prepare(self.optim)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)

        self.accelerator.wait_for_everyone()

    def _init_LMDM(self):
        opt = self.opt

        part_w_dict = None
        if opt.part_w_dict_json:
            part_w_dict = load_json(opt.part_w_dict_json)
        dim_ws = None
        if opt.dim_ws_npy:
            dim_ws = np.load(opt.dim_ws_npy)

        lmdm = LMDM(
            motion_feat_dim=opt.motion_feat_dim,
            audio_feat_dim=opt.audio_feat_dim,
            seq_frames=opt.seq_frames,
            part_w_dict=part_w_dict,   # only for train
            checkpoint=opt.checkpoint,
            device=self.device,
            use_last_frame_loss=opt.use_last_frame_loss,
            use_reg_loss=opt.use_reg_loss,
            dim_ws=dim_ws,
            # ── NEW emotion params ──────────────────────────────────
            use_emotion=getattr(opt, 'use_emotion', False),
            emo_dim=getattr(opt, 'emo_dim', 128),
            hubert_dim=getattr(opt, 'hubert_dim', 1024),
            lambda_emo=getattr(opt, 'lambda_emo', 0.1),
        )

        return lmdm

    def _init_dataloaders(self):
        opt = self.opt

        if opt.dataset_version in ['v2']:
            Stage2Dataset = Stage2DatasetV2
        else:
            raise NotImplementedError()

        # ───────────────────────────────────────────────────────────────────
        # MEAD mode: use per-split data_list JSONs directly (no split_txt needed)
        # ───────────────────────────────────────────────────────────────────
        if opt.mead_train_data_list_json:
            print("[Trainer] MEAD mode: using per-split data_list JSONs.")
            print(f"  train : {opt.mead_train_data_list_json}")
            if opt.mead_val_data_list_json:
                print(f"  val   : {opt.mead_val_data_list_json}")
            if opt.mead_test_data_list_json:
                print(f"  test  : {opt.mead_test_data_list_json}")

            # Shared kwargs for all splits
            shared_ds_kwargs = dict(
                seq_len=opt.seq_frames,
                preload=opt.data_preload,
                cache=opt.data_cache,
                motion_feat_dim=opt.motion_feat_dim,
                motion_feat_start=opt.motion_feat_start,
                motion_feat_offset_dim_se=opt.motion_feat_offset_dim_se,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_emo=opt.use_emo,
                use_sc=opt.use_sc,
                use_last_frame=opt.use_last_frame,
                use_lmk=opt.use_lmk,
                use_cond_end=opt.use_cond_end,
                mtn_mean_var_npy=opt.mtn_mean_var_npy,
                reprepare_idx_map=opt.reprepare_idx_map,
            )

            print("[Trainer] Loading MEAD train dataset...")
            train_dataset = Stage2Dataset(
                data_list_json=opt.mead_train_data_list_json,
                preload_pkl=opt.data_preload_pkl,
                **shared_ds_kwargs,
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )

            val_loader = None
            if opt.mead_val_data_list_json:
                val_batch_size = opt.val_batch_size if opt.val_batch_size > 0 else opt.batch_size
                val_num_workers = opt.val_num_workers if opt.val_num_workers > 0 else opt.num_workers
                print("[Trainer] Loading MEAD val dataset...")
                val_dataset = Stage2Dataset(
                    data_list_json=opt.mead_val_data_list_json,
                    preload_pkl="",
                    **shared_ds_kwargs,
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=val_batch_size,
                    num_workers=val_num_workers,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )

            return train_loader, val_loader

        # ───────────────────────────────────────────────────────────────────
        # Classic mode: flat data_list_json + optional split_txt filtering
        # ───────────────────────────────────────────────────────────────────
        train_dataset = Stage2Dataset(
            data_list_json=opt.data_list_json, 
            seq_len=opt.seq_frames,
            preload=opt.data_preload, 
            cache=opt.data_cache, 
            preload_pkl=opt.data_preload_pkl, 
            motion_feat_dim=opt.motion_feat_dim, 
            motion_feat_start=opt.motion_feat_start,
            motion_feat_offset_dim_se=opt.motion_feat_offset_dim_se,
            use_eye_open=opt.use_eye_open,
            use_eye_ball=opt.use_eye_ball,
            use_emo=opt.use_emo,
            use_sc=opt.use_sc,
            use_last_frame=opt.use_last_frame,
            use_lmk=opt.use_lmk,
            use_cond_end=opt.use_cond_end,
            mtn_mean_var_npy=opt.mtn_mean_var_npy,
            reprepare_idx_map=opt.reprepare_idx_map,
            split_txt=opt.train_split_txt,
            split_strict=opt.split_strict,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = None
        if opt.val_split_txt:
            val_batch_size = opt.val_batch_size if opt.val_batch_size > 0 else opt.batch_size
            val_num_workers = opt.val_num_workers if opt.val_num_workers > 0 else opt.num_workers

            # Validation uses the same underlying features, just filtered by val split.
            val_dataset = Stage2Dataset(
                data_list_json=opt.data_list_json,
                seq_len=opt.seq_frames,
                preload=opt.data_preload,
                cache=opt.data_cache,
                preload_pkl="",  # avoid sharing train preload idx_map implicitly
                motion_feat_dim=opt.motion_feat_dim,
                motion_feat_start=opt.motion_feat_start,
                motion_feat_offset_dim_se=opt.motion_feat_offset_dim_se,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_emo=opt.use_emo,
                use_sc=opt.use_sc,
                use_last_frame=opt.use_last_frame,
                use_lmk=opt.use_lmk,
                use_cond_end=opt.use_cond_end,
                mtn_mean_var_npy=opt.mtn_mean_var_npy,
                reprepare_idx_map=opt.reprepare_idx_map,
                split_txt=opt.val_split_txt,
                split_strict=opt.split_strict,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )

        return train_loader, val_loader
    
    def _init_optim(self):
        opt = self.opt
        # Stage-2 emotion fine-tuning: only optimise new emotion parameters,
        # keeping the pre-trained DiT backbone frozen.
        freeze_backbone = getattr(opt, 'freeze_backbone', False)
        if freeze_backbone and self.LMDM.model.use_emotion:
            print("[Trainer] freeze_backbone=True: optimising emotion modules only.")
            emotion_params = []
            for name, p in self.LMDM.model.named_parameters():
                if 'emotion_encoder' in name or 'emo_adaln' in name \
                        or 'emo_k_proj' in name or 'emo_v_proj' in name \
                        or 'null_emo_embed' in name:
                    emotion_params.append(p)
                else:
                    p.requires_grad_(False)
            optim = Adan(emotion_params, lr=opt.lr, weight_decay=0.02)
        else:
            optim = Adan(self.LMDM.model.parameters(), lr=opt.lr, weight_decay=0.02)
        return optim

    def _init_log(self):
        opt = self.opt
        
        experiment_path = os.path.join(opt.experiment_dir, opt.experiment_name)
        self.error_log_path = os.path.join(experiment_path, 'error')
        self.ckpt_path = os.path.join(experiment_path, 'ckpts')
        self.resume_start_epoch = 1
        self.resume_global_step = 0
        
        if not self.is_main_process:
            return

        # ckpt
        os.makedirs(self.ckpt_path, exist_ok=True)

        # save opt
        opt_pkl = os.path.join(experiment_path, 'opt.pkl')
        dump_pkl(vars(opt), opt_pkl)

        # loss log
        loss_log = os.path.join(experiment_path, 'loss.log')
        self.loss_logger = open(loss_log, 'a')

        self.ckpt_file_list_for_clear = []

    def _get_model_for_state_io(self):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.LMDM.model)
        return self.LMDM.model

    def _get_latest_resume_checkpoint(self):
        if not os.path.exists(self.ckpt_path):
            return ""

        # Prefer a stable "latest" file first.
        latest_path = os.path.join(self.ckpt_path, "resume_last.pt")
        if os.path.exists(latest_path):
            return latest_path

        # Fallback to epoch-stamped files if present.
        resume_files = []
        for name in os.listdir(self.ckpt_path):
            if name.startswith("resume_epoch_") and name.endswith(".pt"):
                resume_files.append(name)
        if not resume_files:
            return ""

        def _extract_epoch(filename):
            # resume_epoch_{N}.pt
            stem = os.path.splitext(filename)[0]
            parts = stem.split("_")
            try:
                return int(parts[-1])
            except Exception:
                return -1

        resume_files.sort(key=_extract_epoch)
        return os.path.join(self.ckpt_path, resume_files[-1])

    def _maybe_resume_training(self):
        opt = self.opt

        resume_path = ""
        if getattr(opt, "resume_checkpoint", ""):
            resume_path = opt.resume_checkpoint
        elif getattr(opt, "resume", False) or getattr(opt, "auto_resume", False):
            resume_path = self._get_latest_resume_checkpoint()

        if not resume_path:
            return
        if not os.path.exists(resume_path):
            if self.is_main_process:
                tqdm.write(f"[RESUME] Checkpoint not found: {resume_path}. Start from epoch 1.")
            return

        checkpoint = torch.load(resume_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"[RESUME] Invalid resume checkpoint format: {resume_path}")

        model_state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint.get("state_dict")
        )
        if model_state is None:
            raise ValueError(f"[RESUME] No model state found in {resume_path}")
        self._get_model_for_state_io().load_state_dict(model_state, strict=False)

        optim_state = checkpoint.get("optimizer_state_dict")
        if optim_state is not None:
            self.optim.load_state_dict(optim_state)

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        last_epoch = int(checkpoint.get("epoch", 0))
        self.resume_start_epoch = last_epoch + 1
        self.resume_global_step = int(checkpoint.get("global_step", 0))

        if self.is_main_process:
            tqdm.write(
                f"[RESUME] Loaded: {resume_path} | "
                f"last_epoch={last_epoch}, global_step={self.resume_global_step}"
            )
            tqdm.write(f"[RESUME] Resuming from epoch {self.resume_start_epoch}")

    def _save_resume_checkpoint(self, epoch, state_dict):
        resume_ckpt = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_val_loss": float(self.best_val_loss),
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optim.state_dict(),
        }
        if hasattr(self, "scheduler") and self.scheduler is not None:
            resume_ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        resume_last_path = os.path.join(self.ckpt_path, "resume_last.pt")
        torch.save(resume_ckpt, resume_last_path)
        # Optional epoch-stamped snapshot for manual rollback/debug.
        resume_epoch_path = os.path.join(self.ckpt_path, f"resume_epoch_{epoch}.pt")
        torch.save(resume_ckpt, resume_epoch_path)

    def _loss_backward(self, loss):
        self.optim.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optim.step()

    def _train_one_step(self, data_dict):
        x          = data_dict["kp_seq"]     # (B, L, kp_dim)
        cond_frame = data_dict["kp_cond"]   # (B, kp_dim)
        cond       = data_dict["aud_cond"]   # (B, L, aud_dim)

        if not self.opt.use_accelerate:
            x          = x.to(self.device)
            cond_frame = cond_frame.to(self.device)
            cond       = cond.to(self.device)

        loss, loss_dict = self.LMDM.diffusion(
            x, cond_frame, cond, t_override=None
        )

        return loss, loss_dict

    def _train_one_epoch(self):
        data_loader = self.train_loader

        DAM = DictAverageMeter()

        self.LMDM.train()
        self.local_step = 0
        for data_dict in tqdm(data_loader, disable=not self.is_main_process):
            self.global_step += 1
            self.local_step += 1

            loss, loss_dict = self._train_one_step(data_dict)
            self._loss_backward(loss)

            if self.is_main_process:
                loss_dict['total_loss'] = loss
                loss_dict_val = {}
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_dict_val[k] = float(v.detach().cpu().item())
                    else:
                        loss_dict_val[k] = float(v)
                DAM.update(loss_dict_val)

        return DAM

    @torch.no_grad()
    def _validate_one_epoch(self):
        if self.val_loader is None:
            return None

        DAM = DictAverageMeter()
        self.LMDM.eval()

        for data_dict in tqdm(self.val_loader, disable=not self.is_main_process, desc="val"):
            loss, loss_dict = self._train_one_step(data_dict)
            if self.is_main_process:
                loss_dict['total_loss'] = loss
                loss_dict_val = {}
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        loss_dict_val[k] = float(v.detach().cpu().item())
                    else:
                        loss_dict_val[k] = float(v)
                DAM.update(loss_dict_val)

        return DAM

    def _show_and_save(self, DAM: DictAverageMeter):
        if not self.is_main_process:
            return
        
        self.LMDM.eval()

        epoch = self.epoch

        def _format_loss_breakdown(loss_avg: dict):
            if not loss_avg:
                return ""

            # Keep total first, then print the rest in sorted order for stable logs.
            keys = []
            if "total_loss" in loss_avg:
                keys.append("total_loss")
            keys.extend(sorted([k for k in loss_avg.keys() if k != "total_loss"]))

            parts = []
            for k in keys:
                v = loss_avg.get(k, None)
                if v is None:
                    continue
                parts.append(f"{k}: {float(v):.6f}")
            return " | ".join(parts)

        # aggregate and log train loss
        train_avg = DAM.average()
        train_breakdown = _format_loss_breakdown(train_avg)
        train_msg = f"Epoch: {epoch}, Global_Steps: {self.global_step}"
        if train_breakdown:
            train_msg += f", | {train_breakdown} |"
        else:
            train_msg += ", | total_loss: N/A |"

        tqdm.write(train_msg)
        print(train_msg, file=self.loss_logger)
        self.loss_logger.flush()

        # optional val log and best checkpoint saving
        val_msg_extra = ""
        if getattr(self, "val_DAM", None) is not None:
            val_avg = self.val_DAM.average()
            val_total = val_avg.get("total_loss", None)
            if val_total is not None:
                val_msg = f"Epoch [{epoch}/{self.opt.epochs}] | Val Loss: {val_total:.6f}"
                print(val_msg)
                print(val_msg, file=self.loss_logger)
                self.loss_logger.flush()

                # save best / last checkpoints
                if val_total < self.best_val_loss:
                    self.best_val_loss = val_total
                    state_dict = self._get_model_for_state_io().state_dict()
                    ckpt = {"model_state_dict": state_dict}
                    best_path = os.path.join(self.ckpt_path, "best_model.pt")
                    torch.save(ckpt, best_path)
                    last_path = os.path.join(self.ckpt_path, "last_model.pt")
                    torch.save(ckpt, last_path)
                    self._save_resume_checkpoint(epoch, state_dict)
                    val_msg_extra = "Best Model Saved"
                    tqdm.write(f"[BEST MODEL SAVED at Epoch {epoch}] val_loss={val_total:.6f}")
                else:
                    # always update last_model.pt for bookkeeping
                    state_dict = self._get_model_for_state_io().state_dict()
                    ckpt = {"model_state_dict": state_dict}
                    last_path = os.path.join(self.ckpt_path, "last_model.pt")
                    torch.save(ckpt, last_path)
                    self._save_resume_checkpoint(epoch, state_dict)
                    val_msg_extra = "Validation did not improve"
                    tqdm.write(f"[NO IMPROVEMENT at Epoch {epoch}] val_loss={val_total:.6f}")

                log_line = f"Epoch [{epoch}/{self.opt.epochs}] | {val_msg_extra}"
                print(log_line)
                print(log_line, file=self.loss_logger)
                self.loss_logger.flush()
        else:
            # no validation loader: keep old per-epoch checkpoint behaviour
            state_dict = self._get_model_for_state_io().state_dict()

            ckpt = {
                "model_state_dict": state_dict,
            }
            ckpt_p = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
            torch.save(ckpt, ckpt_p)
            self._save_resume_checkpoint(epoch, state_dict)
            tqdm.write(f"[MODEL SAVED at Epoch {epoch}] ({len(self.ckpt_file_list_for_clear)})")
            
            # clear model
            if epoch % self.opt.save_ckpt_freq != 0:
                self.ckpt_file_list_for_clear.append(ckpt_p)

            if len(self.ckpt_file_list_for_clear) > 5:
                _ckpt = self.ckpt_file_list_for_clear.pop(0)
                try:
                    os.remove(_ckpt)
                except:
                    traceback.print_exc()
                    self.ckpt_file_list_for_clear.insert(0, _ckpt)

    def _export_best_checkpoint_to_pth(self):
        """Export best checkpoint weights to a fixed .pth path once training ends."""
        if not self.is_main_process:
            return

        candidate_best_paths = [
            os.path.join(self.ckpt_path, "best.pt"),
            os.path.join(self.ckpt_path, "best_model.pt"),
        ]
        best_ckpt_path = next((p for p in candidate_best_paths if os.path.exists(p)), None)
        if best_ckpt_path is None:
            tqdm.write("[POST-TRAIN EXPORT] No best checkpoint found; skipping .pth export.")
            return

        target_pth = os.path.join(
            ".",
            "checkpoints",
            "ditto_pytorch",
            "models",
            "lmdm_v0.4_hubert.pth",
        )
        os.makedirs(os.path.dirname(target_pth), exist_ok=True)

        checkpoint = torch.load(best_ckpt_path, map_location="cpu")
        state_dict = None

        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break

            # If the checkpoint dict itself looks like a state_dict.
            if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint

        if state_dict is None:
            raise ValueError(
                f"Unsupported checkpoint structure in '{best_ckpt_path}'. "
                "Expected a state dict or a dict containing one of "
                "['model_state_dict', 'model', 'state_dict']."
            )

        torch.save(state_dict, target_pth)
        tqdm.write(f"[POST-TRAIN EXPORT] {best_ckpt_path} -> {target_pth}")

    def _train_loop(self):
        print(time.asctime(), 'start ...')

        opt = self.opt

        start_epoch = getattr(self, "resume_start_epoch", 1)
        self.global_step = getattr(self, "resume_global_step", 0)
        self.local_step = 0
        for epoch in trange(start_epoch, opt.epochs + 1, disable=not self.is_main_process):
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            self.epoch = epoch
            DAM = self._train_one_epoch()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.is_main_process:
                self.LMDM.eval()
                self.val_DAM = None
                if self.val_loader is not None and (epoch % max(1, opt.val_freq) == 0):
                    self.val_DAM = self._validate_one_epoch()
                self._show_and_save(DAM)

        self._export_best_checkpoint_to_pth()
        print(time.asctime(), 'done.')

    def train_loop(self):
        try:
            self._train_loop()
        except:
            msg = traceback.format_exc()
            error_msg = f'{time.asctime()} \n {msg} \n'
            print(error_msg)
            t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            logname = f'{t}_rank{self.process_index}_error.log'
            os.makedirs(self.error_log_path, exist_ok=True)
            errorfile = os.path.join(self.error_log_path, logname)
            with open(errorfile, 'a') as f:
                f.write(error_msg)
            print(f'error msg write into {errorfile}')