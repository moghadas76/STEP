import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..dlinear import Model as DLinear

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

import matplotlib.pyplot as plt

def visualize(att):
    def imshow_batch(images, titles=None):
        """
        Display a batch of images in a single figure.

        Parameters:
        - images: A list of NumPy arrays, where each array represents an image.
        - titles: A list of titles for each image. If None, no titles will be displayed.
        """

        num_images = len(images)

        if titles is not None and len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images")

        # Calculate the number of rows and columns for the subplots
        num_rows = int(np.ceil(np.sqrt(num_images)))
        num_cols = int(np.ceil(num_images / num_rows))

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        # Remove any empty subplots
        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

        # Plot each image
        for i in range(num_images):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_images > 1 else axes  # Handle single-image case

            ax.imshow(images[i].cpu().detach().numpy(), cmap='gray')  # Assuming images are grayscale, change cmap if needed
            ax.axis('off')

            if titles is not None:
                ax.set_title(titles[i])
        plt.savefig(f'/home/seyed/PycharmProjects/step/STEP/plots/attention_{str(datetime.datetime.now())}.jpg')
        plt.show()
    imshow_batch(att[:5, ...], [f"att_batch_{batch}" for batch in range(len(att[:5, ...]))])


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
                Forward pass of the GraphAttentionLayer.

                Parameters:
                h (torch.Tensor): Input features of nodes.
                adj (torch.Tensor): Adjacency matrix of the graph.

                Returns:
                torch.Tensor: Output features of nodes.
                """
        qq = h.clone()
        Wh = torch.bmm(h.permute(0, 3, 2, 1).reshape(qq.size(0)*qq.size(3), qq.size(2), qq.size(1)),
                       self.W.unsqueeze(0).repeat(qq.size(0)*qq.size(-1), 1, 1)) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh, bs=qq.size(0))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        self.register_buffer("latest_attention", attention)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('ncvl,nvw->ncwl',(Wh.view(qq.size(0), Wh.size(0)//qq.size(0), Wh.size(1), Wh.size(2)),attention))
        if self.concat:
            #TODO:
            # - Revert me
            # F.elu
            return F.elu(h_prime).permute(0, 3, 2, 1)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, bs=None):
        """

        :param Wh:
        :param bs: Batch size
        :return:
        """
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # import remote_pdb;
        # remote_pdb.set_trace()
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e.view(bs, e.size(0)//bs, e.size(1), e.size(2))[:, -1,:, :].squeeze(1))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphWaveNet(nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, num_nodes, support_len, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):
        """
            kindly note that although there is a 'supports' parameter, we will not use the prior graph if there is a learned dependency graph.
            Details can be found in the feed forward function.
        """
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.fc_his = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))

        receptive_field = 1

        self.supports_len = support_len

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len +=1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        self.receptive_field = receptive_field

    def _calculate_random_walk_matrix(self, adj_mx):
        B, N, N = adj_mx.shape

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
        d = torch.sum(adj_mx, 2)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, input, hidden_states, sampled_adj):
        """feed forward of Graph WaveNet.

        Args:
            input (torch.Tensor): input history MTS with shape [B, L, N, C].
            His (torch.Tensor): the output of TSFormer of the last patch (segment) with shape [B, N, d].
            adj (torch.Tensor): the learned discrete dependency graph with shape [B, N, N].

        Returns:
            torch.Tensor: prediction with shape [B, N, L]
        """

        # reshape input: [B, L, N, C] -> [B, C, N, L]
        input = input.transpose(1, 3)
        # feed forward
        input = nn.functional.pad(input,(1,0,0,0))

        input = input[:, :2, :, :]
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        #
        # ====== if use learned adjacency matrix, then reset the self.supports ===== #
        self.supports = [self._calculate_random_walk_matrix(sampled_adj), self._calculate_random_walk_matrix(sampled_adj.transpose(-1, -2))]

        # calculate the current adaptive adj matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        hidden_states = self.fc_his(hidden_states)        # B, N, D
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        skip = skip + hidden_states
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        return x
        tmp = x.clone()
        x = self.dlinear(x)
        # x = self._prepare_attentional_mechanism_input(x, self.gconv[7].nconv.latest_attention)
        # return x + tmp.unsqueeze(-1)
        return x + tmp
