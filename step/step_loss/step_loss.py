import numpy as np
from torch import nn
from basicts.losses import masked_mae

def step_loss(prediction, real_value, theta, priori_adj, gsl_coefficient, null_val=np.nan):
    # graph structure learning loss
    B, N, N = theta.shape
    theta = theta.view(B, N*N)
    tru = priori_adj.view(B, N*N)
    BCE_loss = nn.BCELoss()
    loss_graph = BCE_loss(theta, tru)
    # prediction loss
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    # final loss
    loss = loss_pred + loss_graph * gsl_coefficient
    return loss


def step_loss_tiny(prediction, real_value,theta=None, priori_adj=None, *arg, **kwargs):
    # graph structure learning loss
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=0.0)
    BCE_loss = nn.BCELoss()
    # loss_graph = BCE_loss(theta, priori_adj)
    loss_graph = 0.0
    # final loss
    return loss_pred + 0.1 * loss_graph
