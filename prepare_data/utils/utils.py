import json
import random
import os
import numpy as np
import torch
import time
import uuid
import pickle


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, pkl):
    with open(pkl, "wb") as fw:
        pickle.dump(obj, fw)
        