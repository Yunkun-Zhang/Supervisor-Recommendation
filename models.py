import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class MetaPath(nn.Module):
    def __init__(self, etypes, out_dim, num_heads, attn_drop=0.5, alpha=0.01, use_minibatch=False):
        super(MetaPath, self).__init__()
        self.etypes = etypes
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.rnn = nn.LSTM(out_dim, num_heads * out_dim)

        self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
        self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        edata = F.embedding(edge_metapath_indices, features)

        _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim

        center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
        a1 = self.attn1(center_node_feat)  # E x num_heads
        a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
        a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1

        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if self.use_minibatch:
            return ret[target_idx]
        else:
            return ret


class MAGNN(nn.Module):
    def __init__(self, num_metapaths, etypes_list, out_dim, num_heads, attn_vec_dim,
                 attn_drop=0.5, use_minibatch=False):
        super(MAGNN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(MetaPath(etypes_list[i],
                                                 out_dim,
                                                 num_heads,
                                                 attn_drop=attn_drop,
                                                 use_minibatch=use_minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer(
                (g, features, type_mask, edge_metapath_indices, target_idx)
            ).view(-1, self.num_heads * self.out_dim)) for g, edge_metapath_indices, target_idx, metapath_layer in
                             zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer(
                (g, features, type_mask, edge_metapath_indices)
            ).view(-1, self.num_heads * self.out_dim)) for g, edge_metapath_indices, metapath_layer in
                             zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h


class LPLayer(nn.Module):
    def __init__(self, num_metapaths_list, etypes_lists, in_dim, out_dim,
                 num_heads, attn_vec_dim, attn_drop=0.5):
        super(LPLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # 0 for paper, 1 for author
        self.paper_layer = MAGNN(num_metapaths_list[0], etypes_lists[0], in_dim,
                                 num_heads, attn_vec_dim, attn_drop, True)
        self.author_layer = MAGNN(num_metapaths_list[1], etypes_lists[1], in_dim,
                                  num_heads, attn_vec_dim, attn_drop, True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_paper = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_author = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_paper.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_author.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific layers
        h_paper = self.paper_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_author = self.author_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        logits_paper = self.fc_paper(h_paper)
        logits_author = self.fc_author(h_author)
        return [logits_paper, logits_author], [h_paper, h_author]


class LP(nn.Module):
    def __init__(self, num_metapaths_list, etypes_lists, feats_dim_list, hidden_dim,
                 out_dim, num_heads, attn_vec_dim, dropout_rate=0.5):
        super(LP, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_lp layers
        self.layer1 = LPLayer(num_metapaths_list, etypes_lists, hidden_dim, out_dim,
                              num_heads, attn_vec_dim, attn_drop=dropout_rate)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_paper, logits_author], [h_paper, h_author] = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))

        return [logits_paper, logits_author], [h_paper, h_author]
