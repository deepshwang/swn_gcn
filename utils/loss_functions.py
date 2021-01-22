import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def apply_margin(outputs, labels, m):
    for i in range(outputs.shape[0]):
        outputs[i][labels[i]] -= m
    return outputs






