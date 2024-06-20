from collections import namedtuple
from typing import Any
from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os, sys
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
sys.path.append(os.path.abspath(__file__ + "/../.."))
from foundation.data.utils import get_dataloader

import numpy as np
import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss), mae_loss

def huber_loss(pred, true, mask_value=None, delta=1.0):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    residual = torch.abs(pred - true)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res)), None
    # lo = torch.nn.SmoothL1Loss()
    # return lo(preds, labels)

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        # print(true[true<1].shape, true[true<0.0001].shape, true[true==0].shape)
        # print(true)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    #if oPNBI>1, pred are bigger than true
    #if oPNBI<1, pred are smaller than true
    #however, this metric is too sentive to small values. Not good!
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def All_Metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        rrse = RRSE_np(pred, true, mask1)
        # corr = 0
        corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae, _ = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape, rrse, corr

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr

class MLP_RL(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, embed_dim, device):
        super(MLP_RL, self).__init__()

        self.ln1 = nn.Linear(dim_in, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, dim_out)

        self.weights_pool_spa = nn.Parameter(torch.randn(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_spa = nn.Parameter(torch.randn(embed_dim, hidden_dim))

        self.weights_pool_tem = nn.Parameter(torch.randn(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_tem = nn.Parameter(torch.randn(embed_dim, hidden_dim))
        self.act = nn.LeakyReLU()
        self.device = device

    def forward(self, eb, time_eb, node_eb):
        eb_out = self.ln1(eb)

        weights_spa = torch.einsum('nd,dio->nio', node_eb, self.weights_pool_spa)
        bias_spa = torch.matmul(node_eb, self.bias_pool_spa)
        out_spa = torch.einsum('btni,nio->btno', eb_out, weights_spa) + bias_spa
        out_spa = self.act(out_spa)

        weights_tem = torch.einsum('btd,dio->btio', time_eb, self.weights_pool_tem)
        bias_tem = torch.matmul(time_eb, self.bias_pool_tem).unsqueeze(-2)
        out_tem = torch.einsum('btni,btio->btno', out_spa, weights_tem) + bias_tem
        out_tem = self.act(out_tem)
        logits = self.ln3(out_tem)
        return logits

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)

class cap_adj(nn.Module):
    def __init__(self, dim, num_nodes, timesteps, embed_dim, embed_dim_spa, mask_R, HS, HT, num_route):
        super(cap_adj, self).__init__()
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.dim = dim
        self.mask_R = mask_R
        self.num_route = num_route
        self.HS = HS
        self.TT = HS * timesteps

        self.ln_p = nn.Linear(dim, dim)
        self.adj = nn.Parameter(torch.randn(embed_dim_spa, HS, num_nodes), requires_grad=True)
        self.LRelu = nn.LeakyReLU()

    def forward(self, x, teb):
        batch_size = x.size(0)
        Pcaps = self.ln_p(x)
        Pcaps_out = squash(Pcaps, dim=-1)
        dadj = torch.einsum('btd,dhn->bthn', teb, self.adj)
        test1 = torch.einsum('bthn,btnd->bthd', dadj.softmax(-2), Pcaps_out)
        Dcaps_in = torch.matmul(squash(test1).unsqueeze(-1).permute(0, 1, 3, 2, 4),
                             Pcaps_out.unsqueeze(-1).permute(0, 1, 3, 2, 4).transpose(-1, -2)).permute(0, 1, 3, 4, 2)
        k_test = Pcaps_out.detach()
        temp_u_hat = Dcaps_in.detach()

        # Routing
        b = torch.zeros(batch_size, self.timesteps, self.HS, self.num_nodes, 1).to('cuda')
        for route_iter in range(self.num_route):
            c = b.softmax(dim=2)
            s = (c * temp_u_hat).sum(-2)
            v = squash(s)
            uv = torch.matmul(v, k_test.transpose(-1, -2)).unsqueeze(-1)
            b += uv

        c = (b + dadj.unsqueeze(-1)).softmax(dim=2)
        return c

class cap(nn.Module):
    def __init__(self, dim, num_nodes, timesteps, embed_dim, embed_dim_spa, HS, HT, num_route):
        super(cap, self).__init__()
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.dim = dim
        self.num_route = num_route
        self.HS = HS
        self.TT = HS * timesteps

        self.ln_p = nn.Linear(dim, dim)
        self.t_adj = nn.Parameter(torch.randn(embed_dim_spa, HT, self.TT), requires_grad=True)
        self.adj = nn.Parameter(torch.randn(embed_dim_spa, HS, num_nodes), requires_grad=True)
        self.weights_spa = nn.Parameter(torch.randn(embed_dim, dim, dim))
        self.bias_spa = nn.Parameter(torch.randn(embed_dim, dim))

        self.LRelu = nn.LeakyReLU()

        mask_template = (torch.linspace(1, timesteps, steps=timesteps)) / 12.
        self.register_buffer('mask_template', mask_template)

    def forward(self, x, node_embeddings, time_eb, teb):
        batch_size = x.size(0)
        Pcaps = self.ln_p(x)
        Pcaps_out = squash(Pcaps, dim=-1)
        dadj = torch.einsum('btd,dhn->bthn', teb, self.adj)
        test1 = torch.einsum('bthn,btnd->bthd', dadj.softmax(-2), Pcaps_out)
        Dcaps_in = torch.matmul(squash(test1).unsqueeze(-1).permute(0, 1, 3, 2, 4),
                             Pcaps_out.unsqueeze(-1).permute(0, 1, 3, 2, 4).transpose(-1, -2)).permute(0, 1, 3, 4, 2)
        k_test = Pcaps_out.detach()
        temp_u_hat = Dcaps_in.detach()

        # Routing
        b = torch.zeros(batch_size, self.timesteps, self.HS, self.num_nodes, 1).to('cuda')
        for route_iter in range(self.num_route):
            c = b.softmax(dim=2)
            s = (c * temp_u_hat).sum(-2)
            v = squash(s)
            uv = torch.matmul(v, k_test.transpose(-1, -2)).unsqueeze(-1)
            b += uv

        c = (b + dadj.unsqueeze(-1)).softmax(dim=2)
        # c_return = b + dadj.unsqueeze(-1)

        s = torch.einsum('bthn,btnd->bthd', c.squeeze(-1), Pcaps_out)

        time_index = self.mask_template.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        hyperEmbeds_spa = s + time_index
        hyperEmbeds_spa = hyperEmbeds_spa.reshape(batch_size, -1, self.dim)

        dynamic_adj = torch.einsum('bd,dhk->bhk', time_eb, self.t_adj)
        hyperEmbeds_tem = self.LRelu(torch.einsum('bhk,bkd->bhd', dynamic_adj, hyperEmbeds_spa))
        retEmbeds_tem = self.LRelu(torch.einsum('bkh,bhd->bkd', dynamic_adj.transpose(-1, -2), hyperEmbeds_tem))
        retEmbeds_tem = retEmbeds_tem.reshape(batch_size, self.timesteps, -1, self.dim) + s

        v = squash(retEmbeds_tem)
        reconstruction = torch.einsum('btnh,bthd->btnd', c.squeeze(-1).transpose(-1, -2), v)

        weights_spatial = torch.einsum('nd,dio->nio', node_embeddings, self.weights_spa)
        bias_spatial = torch.matmul(node_embeddings, self.bias_spa)                 #N, dim_out
        out = torch.einsum('btni,nio->btno', reconstruction, weights_spatial) + bias_spatial  # b, N, dim_out

        return self.LRelu(out + x), c.detach(), dynamic_adj.detach()


class hyperTem(nn.Module):
    def __init__(self, timesteps, num_node, dim_in, dim_out, embed_dim, HT_Tem):
        super(hyperTem, self).__init__()
        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HT_Tem, timesteps), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

    def forward(self, eb, node_embeddings, time_eb):

        adj_dynamics = torch.einsum('nk,kht->nht', node_embeddings, self.adj).permute(1, 2, 0)
        hyperEmbeds = torch.einsum('htn,btnd->bhnd', adj_dynamics, eb)
        retEmbeds = torch.einsum('thn,bhnd->btnd', adj_dynamics.transpose(0, 1), hyperEmbeds)

        weights = torch.einsum('btd,dio->btio', time_eb, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(time_eb, self.bias_pool).unsqueeze(2)                       #N, dim_out
        out = torch.einsum('btni,btio->btno', retEmbeds, weights) + bias     #b, N, dim_out
        return self.act(out + eb)

class hyperSpa(nn.Module):
    def __init__(self, num_node, dim_in, dim_out, embed_dim, HS_Spa):
        super(hyperSpa, self).__init__()
        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HS_Spa, num_node), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

    def forward(self, eb, node_embeddings, time_eb):

        adj_dynamics = torch.einsum('btk,khn->bthn', time_eb, self.adj).permute(1, 2, 0)
        hyperEmbeds = self.act(torch.einsum('bthn,btnd->bthd', adj_dynamics, eb))
        retEmbeds = self.act(torch.einsum('btnh,bthd->btnd', adj_dynamics.transpose(-1, -2), hyperEmbeds))

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                     #N, dim_out
        out = torch.einsum('btni,nio->btno', retEmbeds, weights) + bias     #b, N, dim_out
        return self.act(out + eb)


class time_feature(nn.Module):
    def __init__(self, embed_dim):
        super(time_feature, self).__init__()

        self.ln_day = nn.Linear(1, embed_dim)
        self.ln_week = nn.Linear(1, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb):
        day = self.ln_day(eb[:, :, 0:1])
        week = self.ln_week(eb[:, :, 1:2])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb

class time_feature_spg(nn.Module):
    def __init__(self, embed_dim):
        super(time_feature_spg, self).__init__()

        self.ln_day = nn.Linear(12, embed_dim)
        self.ln_week = nn.Linear(12, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb):
        day = self.ln_day(eb[:, :, 0])
        week = self.ln_week(eb[:, :, 1])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb

class STHCN(nn.Module):
    def __init__(self, args):
        super(STHCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.node_embeddings_spg = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)


        self.hyperTem1 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem2 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem3 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem4 = hyperTem(args.horizon, args.num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)

        self.time_feature1 = time_feature(self.embed_dim)
        self.time_feature1_ = time_feature(self.embed_dim_spa)
        self.time_feature2 = time_feature_spg(self.embed_dim_spa)

        self.cap1 = cap(self.hidden_dim, args.num_nodes, self.horizon, self.embed_dim, self.embed_dim_spa, self.HS, self.HT, self.num_route)
        self.cap2 = cap(self.hidden_dim, args.num_nodes, self.horizon, self.embed_dim, self.embed_dim_spa, self.HS, self.HT, self.num_route)

    def forward(self, source, x_in):
        #source: B, T_1, N, D

        day_index = source[:, :, 0, self.input_base_dim:self.input_base_dim+1]
        week_index = source[:, :, 0, self.input_base_dim+1:self.input_base_dim+2]

        time_eb = self.time_feature1(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        teb = self.time_feature1_(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        time_eb_spg = self.time_feature2(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)

        # print(time_eb.shape, teb.shape, time_eb_spg.shape)

        xt1 = self.hyperTem1(x_in, self.node_embeddings, time_eb)
        x_hyperTem_gnn1, HS1, HT1 = self.cap1(xt1, self.node_embeddings_spg, time_eb_spg, teb)
        xt2 = self.hyperTem2(x_hyperTem_gnn1, self.node_embeddings, time_eb)

        xt3 = self.hyperTem3(xt2, self.node_embeddings, time_eb)
        x_hyperTem_gnn3, HS3, HT3 = self.cap2(xt3, self.node_embeddings_spg, time_eb_spg, teb)
        xt4 = self.hyperTem4(x_hyperTem_gnn3, self.node_embeddings, time_eb)

        return xt4, HS1, HS3


class Hypergraph_encoder(nn.Module):
    def __init__(self, args):
        super(Hypergraph_encoder, self).__init__()
        self.device = args.device
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.lag
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode
        self.scaler_zeros = args.scaler_zeros
        self.interval = args.interval
        self.week_day = args.week_day
        self.mask_ratio = args.mask_ratio
        self.ada_mask_ratio = args.ada_mask_ratio
        self.ada_type = args.ada_type
        self.change_epoch = args.change_epoch
        self.epochs = args.epochs

        self.dim_in_flow = nn.Linear(self.input_base_dim, self.hidden_dim, bias=True)

        self.STHCN_encode = STHCN(args)
        self.hyperguide1 = torch.randn(self.hidden_dim, self.horizon, self.HS, self.num_node).to('cuda')
        self.MLP_RL = MLP_RL(args.input_base_dim, self.HS, self.hidden_dim, self.embed_dim, self.device)
        self.teb4mask = time_feature(self.embed_dim)
        self.neb4mask = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.act = nn.LeakyReLU()

    def forward(self, source, label, epoch=None):
        if self.mode == 'pretrain':
            if epoch <= self.change_epoch:
                # random mask sta2
                mask_random_init = torch.rand_like(source[..., 0:self.input_base_dim].reshape(-1)).to('cuda')
                _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)
                mask_num = int(mask_random_init.shape[0] * self.mask_ratio)
                max_idx = max_idx_random[:mask_num]  # NYC_TAXI
                mask_random = torch.ones_like(max_idx_random)
                mask_random = mask_random.scatter_(0, max_idx, 0)
                mask_random = mask_random.reshape(-1, self.horizon, self.num_node, self.input_base_dim)
                final_mask = mask_random

                # get the HS first
                day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
                week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
                time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
                guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)

                # get the classification label
                softmax_guide_weight = F.softmax(guide_weight, dim=-1)
            else:
                ### intra-class  inter-class

                # get the HS first
                day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
                week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
                time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
                guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)

                # get the classification label
                softmax_guide_weight = F.softmax(guide_weight, dim=-1)
                max_value, max_idx_all = torch.sort(softmax_guide_weight, dim=-1, descending=True)
                label_c = max_idx_all[..., 0]    # [batch_size, time_steps, num_node]

                # calculate number of random mask and adaptive mask
                train_process = ((epoch - self.change_epoch) / (self.epochs - self.change_epoch)) * self.ada_mask_ratio
                if train_process > 1 :
                    train_process = 1
                mask_num_sum = int(source[:, :, :, 0].reshape(-1).shape[0] * self.mask_ratio)
                adaptive_mask_num = int(mask_num_sum * train_process)
                random_mask_num = mask_num_sum - adaptive_mask_num

                ### adaptive mask
                # random choose mask class until the adaptive_mask_num<=select_num
                list_c = list(range(0, self.HS))
                random.shuffle(list_c)
                select_c = torch.zeros_like(label_c).to(self.device)
                select_d = torch.zeros_like(label_c).to(self.device)
                select_f = torch.zeros_like(label_c).to(self.device)
                select_num = 0
                i = 0

                if self.ada_type == 'all':
                    while select_num < adaptive_mask_num:
                        select_c[label_c == list_c[i]] = 1
                        select_num = torch.sum(select_c)
                        i = i + 1
                    if i >= 2:
                        for k in range(i-1):
                            select_d[label_c == list_c[k]] = 1
                            adaptive_dnum = torch.sum(select_d)
                        select_f[label_c == list_c[i-1]] = 1
                    else:
                        adaptive_dnum = 0
                        select_f = select_c.clone()
                else:
                    while select_num < adaptive_mask_num:
                        select_c[label_c == list_c[i]] = 1
                        select_num = torch.sum(select_c)
                        i = i + 1
                    adaptive_dnum = 0
                    select_f = select_c.clone()

                # randomly choose top adaptive_mask_num to mask
                select_f = select_f.reshape(-1, self.horizon*self.num_node).reshape(-1)
                select_d = select_d.reshape(-1, self.horizon*self.num_node).reshape(-1)
                mask_adaptive_init = torch.rand_like(source[..., 0:1].reshape(-1)).to('cuda')
                mask_adaptive_init = select_f * mask_adaptive_init
                _, max_idx_adaptive = torch.sort(mask_adaptive_init, dim=0, descending=True)

                select_idx_adaptive = max_idx_adaptive[:(adaptive_mask_num-adaptive_dnum)]

                mask_adaptive = torch.ones_like(max_idx_adaptive)
                mask_adaptive = mask_adaptive.scatter_(0, select_idx_adaptive, 0)
                mask_adaptive = mask_adaptive * (1-select_d)

                # random mask
                mask_random_init = torch.rand_like(source[..., 0:1].reshape(-1)).to('cuda')
                mask_random_init = mask_adaptive * mask_random_init
                _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)

                select_idx_random = max_idx_random[:random_mask_num]
                mask_random = torch.ones_like(max_idx_random)
                mask_random = mask_random.scatter_(0, select_idx_random, 0)
                mask_random = mask_random.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon, self.num_node)

                # final_mask
                mask_adaptive = mask_adaptive.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon, self.num_node)
                final_mask = (mask_adaptive * mask_random).unsqueeze(-1)
                if self.input_base_dim != 1:
                    final_mask = final_mask.repeat(1, 1, 1, self.input_base_dim)

            final_mask = final_mask.detach()
            mask_source = final_mask * source[..., 0:self.input_base_dim]
            mask_source[final_mask==0] = self.scaler_zeros
            x_flow_eb = self.dim_in_flow(mask_source)
        else:
            x_flow_eb = self.dim_in_flow(source[..., 0:self.input_base_dim])
        x_flow_encode, HS1, _ = self.STHCN_encode(source, x_flow_eb)

        if self.mode == 'pretrain':
            HS_cat = HS1.squeeze(-1).transpose(-1, -2)
            return x_flow_encode, final_mask[..., :self.input_base_dim], softmax_guide_weight, HS_cat
        else:
            return x_flow_encode

class Hypergraph_decoder(nn.Module):
    def __init__(self, args):
        super(Hypergraph_decoder, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode

        self.time_feature1_ = time_feature(self.embed_dim_spa)
        self.time_feature2_ = time_feature(self.embed_dim_spa)

        self.STHCN_decode = STHCN(args)
        self.dim_flow_out = nn.Linear(self.hidden_dim, self.input_base_dim, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, source, flow_encode_eb):
        flow_decode, HS1, HS2 = self.STHCN_decode(source, flow_encode_eb)
        flow_out = self.dim_flow_out(flow_decode)
        return flow_out, flow_decode
    

from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger


mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs", save_dir="ml-runs", log_model=True)
logger = TensorBoardLogger("tb_logs", name="GPTST_Model")

class GPTST_Model(nn.Module):
    def __init__(self, args):
        super(GPTST_Model, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode
        self.model = args.model

        self.encoder = Hypergraph_encoder(args)
        self.decoder = Hypergraph_decoder(args)

    def forward_pretrain(self, source, label, batch_seen=None, epoch=None):
        flow_encode_eb, mask, probability, HS1 = self.encoder(source, label, epoch)
        flow_out, flow_decode = self.decoder(source, flow_encode_eb)
        return flow_out, flow_decode, 1-mask, probability, HS1

    def forward_fune(self, source, label):
        flow_encode_eb = self.encoder(source, label)
        return flow_encode_eb, flow_encode_eb, flow_encode_eb, flow_encode_eb, flow_encode_eb

    def forward(self, source, label, batch_seen=None, epoch=None):
        if self.mode == 'pretrain':
            return self.forward_pretrain(source, label, batch_seen, epoch)
        else:
            return self.forward_fune(source, label)


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss), mae_loss

def scaler_mae_loss(args, scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss


class GPTST(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args[0]
        self.scaler = args[1]
        self.max_epochs = 101
        self.model = GPTST_Model(args=args[0])

    def forward(self, source, label, batch_seen=None, epoch=None):
        return self.model(source, label, batch_seen, epoch)

    def training_step(self, batch, batch_idx):
        source, label = batch
        loss = scaler_mae_loss(self.args, scaler_data, mask_value=0.0)
        loss_kl = nn.KLDivLoss(reduction='sum').to(self.args.device)
        out, _, mask, probability, eb = self.model(source, label, batch_idx, self.current_epoch)
        loss_flow, _ = loss(out, label[..., :self.args.output_dim], mask)
        if self.current_epoch > self.args.change_epoch :
            loss_s = loss_kl(probability.log(), eb) * 0.1
            loss_val = loss_flow + loss_s
        else:
            loss_val = loss_flow
        print("train_loss: ", loss_val)
        self.log('train_loss', loss_val)
        return loss_val
    
    def validation_step(self, batch, batch_idx):
        source, label = batch
        loss = scaler_mae_loss(self.args, scaler_data, mask_value=0.0)
        loss_kl = nn.KLDivLoss(reduction='sum').to(self.args.device)
        out, _, mask, probability, eb = self.model(source, label, batch_idx, self.current_epoch)
        loss_flow, _ = loss(out, label[..., :self.args.output_dim], mask)
        if self.current_epoch > self.args.change_epoch :
            loss_s = loss_kl(probability.log(), eb) * 0.1
            loss_val = loss_flow + loss_s
        else:
            loss_val = loss_flow
        print("val_loss: ", loss_val)
        self.log('valid_loss', loss_val, sync_dist=True)
        return loss_val
    
    def test_step(self, batch, batch_idx) -> Any:
        model.eval()
        data, gt = batch
        y_pred = []
        y_true = []
        with torch.no_grad():
            data = data[..., :self.args.input_base_dim + self.args.input_extra_dim]
            # label = target[..., :args.input_base_dim + args.input_extra_dim]
            if self.args.mode == 'pretrain':
                output, _, mask, _, _ = self.model(data, None, None, self.args.epochs)
                label = data[..., :self.args.output_dim]
                y_true.append(label*mask)
                y_pred.append(output*mask)
            else:
                output, _, mask, _, _ = model(data, label=None)
                label = gt[..., :self.args.output_dim]
                y_true.append(label)
                y_pred.append(output)
            y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0))
            # if args.real_value:
            #     y_pred = torch.cat(y_pred, dim=0)
            # else:
            y_pred = self.scaler.inverse_transform(torch.cat(y_pred, dim=0))
            # np.save('./{}_true.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_true.cpu().numpy())
            # np.save('./{}_pred.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_pred.cpu().numpy())
            for t in range(y_true.shape[1]):
                mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                    self.args.mae_thresh, self.args.mape_thresh)
                self.log(f'valid_mae_{5*(t+1)}_hor', mae, sync_dist=True)
                self.log(f'valid_rmse_{5*(t+1)}_hor', rmse, sync_dist=True)
                self.log(f'valid_mape_{5*(t+1)}_hor', mape*100, sync_dist=True)
                self.log(f'valid_corr_{5*(t+1)}_hor', corr, sync_dist=True)
            # self.log(f'valid_mae_{5*(t+1)}_hor', mae, sync_dist=True)
            # self.log(f'valid_mae_{5*(t+1)}_hor', mae, sync_dist=True)            
            # logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, CORR:{:.4f}%".format(
            #     t + 1, mae, rmse, mape*100, corr))

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        consine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0)
        return {'optimizer': optim, 'lr_scheduler': consine_scheduler}


import argparse
# import numpy as np
import configparser
# import pandas as pd

def parse_args(device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):


    # parser
    args = argparse.ArgumentParser(prefix_chars='-', description='pretrain_arguments')
    args.add_argument('-dataset', default='METR_LA', type=str, required=True)
    args.add_argument('-mode', default='ori', type=str, required=True)
    args.add_argument('-device', default=device, type=str, help='indices of GPUs')
    args.add_argument('-model', default='TGCN', type=str)
    args.add_argument('-cuda', default=True, type=bool)

    args_get, _ = args.parse_known_args()

    # get configuration
    config_file = '/home/seyed/PycharmProjects/step/STEP/foundation/pems.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    args.add_argument('-val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('-test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('-lag', default=config['data']['lag'], type=int)
    args.add_argument('-horizon', default=config['data']['horizon'], type=int)
    args.add_argument('-num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('-tod', default=config['data']['tod'], type=eval)
    args.add_argument('-normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('-column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('-default_graph', default=config['data']['default_graph'], type=eval)
    # model
    args.add_argument('-input_base_dim', default=config['model']['input_base_dim'], type=int)
    args.add_argument('-input_extra_dim', default=config['model']['input_extra_dim'], type=int)
    args.add_argument('-output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('-embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('-embed_dim_spa', default=config['model']['embed_dim_spa'], type=int)
    args.add_argument('-hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('-HS', default=config['model']['HS'], type=int)
    args.add_argument('-HT', default=config['model']['HT'], type=int)
    args.add_argument('-HT_Tem', default=config['model']['HT_Tem'], type=int)
    args.add_argument('-num_route', default=config['model']['num_route'], type=int)
    args.add_argument('-mask_ratio', default=config['model']['mask_ratio'], type=float)
    args.add_argument('-ada_mask_ratio', default=config['model']['ada_mask_ratio'], type=float)
    args.add_argument('-ada_type', default=config['model']['ada_type'], type=str)
    # train
    args.add_argument('-loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('-seed', default=config['train']['seed'], type=int)
    args.add_argument('-batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('-epochs', default=config['train']['epochs'], type=int)
    args.add_argument('-lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('-lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('-lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('-lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('-early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('-early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('-change_epoch', default=config['train']['change_epoch'], type=int)
    args.add_argument('-up_epoch', default=config['train']['up_epoch'], type=str)
    args.add_argument('-grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('-max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('-debug', default=config['train']['debug'], type=eval)
    args.add_argument('-real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')
    args.add_argument('-seed_mode', default=config['train']['seed_mode'], type=eval)
    args.add_argument('-xavier', default=config['train']['xavier'], type=eval)
    args.add_argument('-load_pretrain_path', default=config['train']['load_pretrain_path'], type=str)
    args.add_argument('-save_pretrain_path', default=config['train']['save_pretrain_path'], type=str)
    # test
    args.add_argument('-mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('-mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    args.add_argument('-log_dir', default='./', type=str)
    args.add_argument('-log_step', default=config['log']['log_step'], type=int)
    args.add_argument('-plot', default=config['log']['plot'], type=eval)
    args, _ = args.parse_known_args()
    return args

argus = parse_args()
# Initialize model

print("!!!!!!!   RANK", int(os.getenv("RANK", -1)))
# Setup data
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week, scaler_holiday = get_dataloader(argus,
                                                               normalizer="std",
                                                               tod=False, dow=False,
                                                               weather=False, single=False)
argus.scaler_zeros = scaler_data.transform(0)
argus.scaler_zeros_day = scaler_day.transform(0)
argus.scaler_zeros_week = scaler_week.transform(0)
model = GPTST(argus, scaler_data)
# args.scaler_zeros_holiday = scaler_holiday.transform(0)

# Initialize a trainer
# trainer = pl.Trainer(devices=2, strategy='ddp_find_unused_parameters_true', logger=[mlf_logger, logger], max_epochs=1001)
trainer = pl.Trainer(devices=1, strategy='auto', logger=[mlf_logger, logger], max_epochs=1001)
# Train the model

CHECKPOINT = "/home/seyed/PycharmProjects/step/STEP/ml-runs/563117307583575659/d598ad3cfb7643b58c0edd54343d4380/artifacts/model/checkpoints/epoch=999-step=84000/epoch=999-step=84000.ckpt"
trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=None)
trainer.test(model, test_loader)