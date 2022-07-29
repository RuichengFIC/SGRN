from torch import nn
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch.nn import Module, Parameter
import math 
from utils import *


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)  # expansion
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs) # reduction
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask= None, src_key_padding_mask= None):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # residual
        src = self.norm1(src) if self.d_model != 1 else src  # conditional layer-norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 2-layer linaer transformation
        src = src + self.dropout2(src2)  # residual & transformation
        src = self.norm2(src) if self.dim_feedforward != 1 else src  # conditional layer-norm
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

        
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        return output

    
class SCD_cell(nn.Module):
    def __init__(self, bottom_size, hidden_size, top_size, a, last_layer):
        super(SCD_cell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.a = a
        self.last_layer = last_layer
        self.U_11 = Parameter(torch.zeros(4 * self.hidden_size + 1, self.hidden_size))
        if not self.last_layer:
            self.U_21 = Parameter(torch.zeros(4 * self.hidden_size + 1, self.top_size))
        self.W_01 = Parameter(torch.zeros(4 * self.hidden_size + 1, self.bottom_size))
        self.bias = Parameter(torch.zeros(4 * self.hidden_size + 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        s_recur_ = torch.mm(self.U_11, h)
        s_recur = (1 - z.expand_as(s_recur_)) * s_recur_
        
        if not self.last_layer:
            s_topdown_ = torch.mm(self.U_21, h_top)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = Variable(torch.zeros(s_recur.size(),device = c.device), requires_grad=False)
        s_bottomup_ = torch.mm(self.W_01, h_bottom)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_

        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = torch.sigmoid(f_s[0:self.hidden_size, :])  # hidden_size * batch_size
        i = torch.sigmoid(f_s[self.hidden_size:self.hidden_size*2, :])
        o = torch.sigmoid(f_s[self.hidden_size*2:self.hidden_size*3, :])
        g = torch.tanh(f_s[self.hidden_size*3:self.hidden_size*4, :])
        z_hat = hard_sigm(self.a, f_s[self.hidden_size*4:self.hidden_size*4+1, :])

        one = Variable(torch.ones(f.size(),device = c.device), requires_grad=False)
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)

        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = z * o * torch.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * torch.tanh(c_new)

        z_new = z_hat


        return h_new, c_new, z_new


class SubsequencAware_LSTM(nn.Module):
    
    def __init__(self, a, input_size, size_list):
        super(SubsequencAware_LSTM, self).__init__()
        self.a = a
        self.input_size = input_size
        self.size_list = size_list

        self.cell_1 = SCD_cell(self.input_size, self.size_list[0], self.size_list[1], self.a, False)
        self.cell_2 = SCD_cell(self.size_list[0], self.size_list[1], None, self.a, True)
        self.drop = torch.nn.Dropout(p=0.5)
        
        self.weight = nn.Linear(size_list[0]+size_list[1], 2)
        self.embed_out1 = nn.Linear(size_list[0], input_size)
        self.embed_out2 = nn.Linear(size_list[1], input_size)
        self.relu = nn.ReLU()

    def forward(self, inputs, hidden = None):
        # inputs.size = (batch_size, time steps, embed_size/input_size)
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        if hidden == None:
            h_t1 = Variable(torch.zeros(self.size_list[0], batch_size, dtype=inputs.dtype,  device= inputs.device), requires_grad=False)
            c_t1 = Variable(torch.zeros(self.size_list[0], batch_size, dtype=inputs.dtype, device= inputs.device), requires_grad=False)
            z_t1 = Variable(torch.zeros(1, batch_size, dtype=inputs.dtype, device= inputs.device), requires_grad=False)
            h_t2 = Variable(torch.zeros(self.size_list[1], batch_size,dtype=inputs.dtype,  device= inputs.device), requires_grad=False)
            c_t2 = Variable(torch.zeros(self.size_list[1], batch_size, dtype=inputs.dtype, device= inputs.device), requires_grad=False)
            z_t2 = Variable(torch.zeros(1, batch_size, dtype=inputs.dtype, device= inputs.device), requires_grad=False)
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = hidden
        z_one = Variable(torch.ones(1, batch_size, dtype=inputs.dtype, device= inputs.device), requires_grad=False)

        h_1 = []
        h_2 = []
        z_1 = []
        z_2 = []
        for t in range(time_steps):
            h_t1, c_t1, z_t1 = self.cell_1(c=c_t1, h_bottom=inputs[:, t, :].t(), h=h_t1, h_top=h_t2, z=z_t1, z_bottom=z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_t2, z_bottom=z_t1)  # 0.01s used
            h_1 += [h_t1.t()]
            h_2 += [h_t2.t()]
            z_1 += [z_t1.t()]
            z_2 += [z_t2.t()]

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        
        h_1, h_2, z_1, z_2, hidden = torch.stack(h_1, dim=1), torch.stack(h_2, dim=1), torch.stack(z_1, dim=1), torch.stack(z_2, dim=1), hidden
        
        result = self.drop(h_2  * z_2)         
        return result


class MRS(nn.Module):
    def __init__(self, ipt_dim, hid_dim, lr_num= 2):
        super(MRS, self).__init__()
        self.lstm= nn.LSTM(input_size = ipt_dim , hidden_size=hid_dim, num_layers=lr_num, bidirectional=True,batch_first=True) 
        self.lstm_sim= nn.LSTM(input_size = ipt_dim * 2, hidden_size=hid_dim, num_layers=lr_num, bidirectional=True,batch_first=True) 

        self.Leakyrelu = torch.nn.LeakyReLU()
        
    def forward(self, stock, market):

        opts, _     = self.lstm.forward(stock)
        opts_sim, _  = self.lstm_sim.forward(torch.cat([stock,market],axis = -1))

        return opts - opts_sim, opts_sim
