from torch import nn
import torch
from layers import *

class SCD(nn.Module):

    name = "SCD"

    def __init__(self, ipt_dim, hid_dim, num_class, lr_num=2, bi_direc=True, batch_first=True,
                 use_MRS=False):
        super(SCD, self).__init__()
        self.use_MRS = use_MRS
        if self.use_MRS:
            self.mrs = MRS(ipt_dim=ipt_dim, hid_dim=hid_dim)
        else:
            self.mrs = nn.LSTM(input_size=ipt_dim, hidden_size=hid_dim, num_layers=lr_num,
                               bidirectional=bi_direc, batch_first=True)

        self.sa_LSTM = SubsequencAware_LSTM(1, 2 * hid_dim, [2 * hid_dim, 2 * hid_dim])

        encoder_layer = TransformerEncoderLayer(d_model=2 * hid_dim,
                                                nhead=2,
                                                dim_feedforward=2 * hid_dim,
                                                batch_first=batch_first)

        self.transformer_encoder = TransformerEncoder(encoder_layer, lr_num)

        self.mlp = nn.Linear(in_features=hid_dim * 2, out_features=num_class, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        if self.use_MRS:
            x_stock = x[:, :, :1]
            x_market = x[:, :, 1:]
            stock_specific, market_induced = self.mrs(x_stock, x_market)
            opts = stock_specific
        else:
            # bilstm
            opts, _ = self.mrs.forward(x)

        saLSTM_opts = self.sa_LSTM(opts)
        h = self.transformer_encoder(saLSTM_opts)
        # mlp
        result = self.mlp.forward(h[:, -1, :])
        result = self.softmax(result)

        return result
