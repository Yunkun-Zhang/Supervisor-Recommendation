import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from models import LP
from utils.meta_path import load_mp
from utils.tools import EarlyStopping, IndexGenerator, parse_minibatch
import argparse

num_ntype = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etype_lists = [[[0, 1], [2, 3], [2, 5, 4, 3]],
               [[1, 0], [5, 6], [4, 3, 2, 5]],
               [[6, 6]]]
use_masks = [[True, False, False], [True, False, False], [False]]
no_masks = [[False] * 3, [False] * 3, [False]]
num_paper = 197378
num_author = 39791
num_field = 110249
expected_metapaths = [
        [(0, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
        [(1, 0, 1), (1, 2, 1), (1, 2, 0, 2, 1)],
        [(2, 2, 2)]
    ]


def run(hidden_dim, num_heads, attn_vec_dim, batch_size, epochs, patience, neighbor_samples, save):
    adjlists, edge_metapath_indices_list, type_mask, pos, neg = load_mp()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    # feats_type = 1
    for i in range(num_ntype):
        dim = 10
        num_nodes = (type_mask == i).sum()
        in_dims.append(dim)
        features_list.append(torch.zeros((num_nodes, 10)).to(device))

    train_pos_author_field = pos['train_pos_author_field']
    train_pos_paper_field = pos['train_pos_paper_field']
    val_pos_author_field = pos['val_pos_author_field']
    val_pos_paper_field = pos['val_pos_paper_field']
    test_pos_author_field = pos['test_pos_author_field']
    test_pos_paper_field = pos['test_pos_paper_field']
    train_neg_author_field = neg['train_pos_author_field']
    train_neg_paper_field = neg['train_pos_paper_field']
    val_neg_author_field = neg['val_pos_author_field']
    val_neg_paper_field = neg['val_pos_paper_field']
    test_neg_author_field = neg['test_pos_author_field']
    test_neg_paper_field = neg['test_pos_paper_field']

    auc_list = []
    ap_list = []
    net = LP([3, 3, 1], etype_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, dropout_rate)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   save_path=f'checkpoint/checkpoint_{save}.pt')
    train_pos_author_field_idx_generator = IndexGenerator(batch_size=batch_size, num_data=len(train_pos_author_field))
    train_pos_paper_field_idx_generator = IndexGenerator(batch_size=batch_size, num_data=len(train_pos_paper_field))
    val_author_field_idx_generator = IndexGenerator(batch_size=batch_size, num_data=len(val_pos_author_field),
                                                    shuffle=False)
    val_paper_field_idx_generator = IndexGenerator(batch_size=batch_size, num_data=len(val_pos_paper_field),
                                                   shuffle=False)

    for epoch in range(epochs):
        for itr in tqdm.trange(train_pos_author_field_idx_generator.num_iterations()):
            train_pos_author_field_idx_batch = train_pos_author_field_idx_generator.next()
            train_pos_author_field_idx_batch.sort()
            train_pos_paper_field_idx_batch = train_pos_paper_field_idx_generator.next()
            train_pos_paper_field_idx_batch.sort()
            train_pos_author_field_batch = train_pos_author_field[train_pos_author_field_idx_batch].tolist()
            train_pos_paper_field_batch = train_pos_paper_field[train_pos_paper_field_idx_batch].tolist()
            train_neg_author_field_idx_batch = np.random.choice(len(train_neg_author_field),
                                                                len(train_pos_author_field_idx_batch))
            train_neg_paper_field_idx_batch = np.random.choice(len(train_neg_paper_field),
                                                               len(train_pos_paper_field_idx_batch))
            train_neg_author_field_idx_batch.sort()
            train_neg_paper_field_idx_batch.sort()
            train_neg_author_field_batch = train_neg_author_field[train_neg_author_field_idx_batch].tolist()
            train_neg_paper_field_batch = train_neg_paper_field[train_neg_paper_field_idx_batch].tolist()

            train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_pos_author_field_batch, device, neighbor_samples,
                use_masks, num_paper)
            train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_neg_author_field_batch, device, neighbor_samples,
                no_masks, num_paper)

            [pos_embedding_paper, pos_embedding_author], _ = net((
                train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
            [neg_embedding_paper, neg_embedding_author], _ = net((
                train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
            pos_embedding_paper = pos_embedding_paper.view(-1, 1, pos_embedding_paper.shape[1])
            pos_embedding_author = pos_embedding_author.view(-1, pos_embedding_author.shape[1], 1)
            neg_embedding_paper = neg_embedding_paper.view(-1, 1, neg_embedding_paper.shape[1])
            neg_embedding_author = neg_embedding_author.view(-1, neg_embedding_author.shape[1], 1)
            pos_out = torch.bmm(pos_embedding_paper, pos_embedding_author)
            neg_out = -torch.bmm(neg_embedding_paper, neg_embedding_author)
            train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', default=8, type=int)