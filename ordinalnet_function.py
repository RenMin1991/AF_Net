# -*- coding: utf-8 -*-
'''
    translation layer, ordinal layer, iris padding layer
    Ren Min
    20181023
'''

import torch
import torch.nn.functional as F
import pdb


def translation(x, trans_col, trans_row):
    # trans_col>0: move towards right
    # trans_row>0: move towards down

    if trans_col != 0:
	x_a = x[:,:,:,-trans_col:]
	x_b = x[:,:,:,:-trans_col]
        x_col = torch.cat((x_a,x_b),dim=3)
    else:
        x_col = x
    if trans_row != 0:
        x_col_a = x_col[:, :, -trans_col:, :]
        x_col_b = x_col[:, :, :-trans_col, :]
        x_col_row = torch.cat((x_col_a,x_col_b),dim=2)
    else:
        x_col_row = x_col
    return x_col_row


def translation3d(x, trans_col, trans_row):
    # trans_col>0: move towards right
    # trans_row>0: move towards down
    if trans_col != 0:
	x_a = x[:,:,-trans_col:]
	x_b = x[:,:,:-trans_col]
        x_col = torch.cat((x_a,x_b),dim=2)
    else:
        x_col = x
    if trans_row != 0:
        x_col_a = x_col[:, -trans_col:, :]
        x_col_b = x_col[:, :-trans_col, :]
        x_col_row = torch.cat((x_col_a,x_col_b),dim=1)
    else:
        x_col_row = x_col
    return x_col_row


def Di_ordinal(x_1, x_2, lamb):
    m = x_1 - x_2
    m = lamb * m
    return F.tanh(m)


def Ti_ordinal(x_1, x_2, x_3, lamb):
    m = x_1 - 2*x_2 + x_3
    m = lamb * m
    return F.tanh(m)


def iris_padding(x, num_col, num_row):
    x_padding_col = x[:, :, :, :num_col]
    x = torch.cat((x,x_padding_col), dim=3)
    x_padding_row = x[:, :, -num_row:, :]
    x_padding_row = x_padding_row * 0.
    x = torch.cat((x,x_padding_row), dim=2)
    return x

