from torch.autograd import Variable, Function
import torch
import torch.nn as nn
from torch import functional as F
from torch.nn import Module
import numpy as np


def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output



class bound(Function):
    @staticmethod
    def forward(ctx, x):
        # forward : x -> output
        result = x > 0.5
        ctx.save_for_backward(result)
        return result.double()
    @staticmethod
    def backward(ctx, grad_output):
        # backward: output_grad -> x_grad
        result, = ctx.saved_tensors
        x_grad = grad_output * result

        return x_grad

def distort_seq(seq, prob = 0.5, noise_level = 0.1):
    """
    Seq: nparray
    """
    new_list = []
    length = len(seq)
    for i in range(length):
        num = seq[i]
        if np.random.rand() > 1 - prob: # 是否变动
            if np.random.rand() > 0.3:     ##不丢弃
                op = np.random.randint(0,3,1)[0]
                ratio = np.random.rand() * prob + 1
                if op == 0:
                    #插值
                    new_list.append(seq[i])
                    if i != length - 1:
                        new_list.append((seq[i] + seq[i+1])/2)
                if op == 1:
                    #放缩:
                    new_list.append(seq[i] * ratio) 
                if op == 2:
                    new_list.append(seq[i] * 0.5)
                    if i != length - 1:
                        new_list.append((seq[i] * ratio  + seq[i+1] * ratio) /2)
                        seq[i+1]  = seq[i+1]* ratio  
        else:
            new_list.append(num)
    new_list = np.array(new_list)      
    new_list = new_list + (np.random.random(len(new_list)) - 0.5) * 2 * noise_level
    return new_list 


def reset_patameters(model):
    for layers in model.children():
        for layer in layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def write_log(log_file, string, mode = "a"):
    with open(log_file, mode) as fo:
        fo.write(string + "\n")