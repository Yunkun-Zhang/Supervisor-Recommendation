import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from models import LP
from utils.meta_path import load_mp
from utils.tools import EarlyStopping, IndexGenerator, parse_minibatch
import argparse

num_ntype = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etype_lists = [[[0, 1], [2, 3], [2, 5, 4, 3]],
               [[1, 0], [4, 5], [4, 3, 2, 5]],
               [[None, None]]]
use_masks_af = [[0, 0, 3], [0, 1, 1], [0]]
use_masks_pf = [[0, 1, 1], [0, 0, 3], [0]]
no_masks = [[0] * 3, [0] * 3, [0]]
num_paper = 197378
num_author = 39791
num_field = 110249
expected_metapaths = [
    [(0, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 2, 0, 2, 1)],
    [(2, 2, 2)]
]


def run(hidden_dim, num_heads, attn_vec_dim, rnn_type, batch_size, epochs, patience, neighbor_samples):
    adjlists, edge_metapath_indices_lists, type_mask, pos_af, neg_af, pos_pf, neg_pf = load_mp('data/preprocessed')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    # feats_type = 1
    for i in range(num_ntype):
        dim = 10
        num_nodes = (type_mask == i).sum()
        in_dims.append(dim)
        features_list.append(torch.zeros((num_nodes, 10)).to(device))

    train_pos_author_field_all = pos_af['train_pos_author_field']
    train_pos_paper_field_all = pos_pf['train_pos_paper_field']
    train_neg_author_field = neg_af['train_neg_author_field']
    train_neg_paper_field = neg_pf['train_neg_paper_field']

    # define model
    net = LP([3, 3, 1], 6, etype_lists, in_dims, hidden_dim, hidden_dim,
             num_heads, attn_vec_dim, rnn_type, dropout_rate)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   save_path=f'checkpoint/checkpoint.pt')

    num_af, num_pf = len(train_pos_author_field_all), len(train_pos_paper_field_all)
    if num_af < num_pf:
        complementary_af = train_pos_author_field_all[np.random.choice(num_af, num_pf - num_af, replace=True)]
        train_pos_author_field_all = np.concatenate((train_pos_author_field_all, complementary_af))
    else:
        complementary_pf = train_pos_paper_field_all[np.random.choice(num_pf, num_af - num_pf, replace=True)]
        train_pos_paper_field_all = np.concatenate((train_pos_paper_field_all, complementary_pf))
    train_steps = max(num_af, num_pf)

    # run
    for epoch in range(epochs):
        # adjust sample counts
        """num_af, num_pf = len(train_pos_author_field_all), len(train_pos_paper_field_all)
        if num_af < num_pf:
            train_pos_paper_field = train_pos_paper_field_all[np.random.choice(num_pf, num_af, replace=False)]
            train_pos_author_field = train_pos_author_field_all.copy()
        else:
            train_pos_author_field = train_pos_author_field_all[np.random.choice(num_af, num_pf, replace=False)]
            train_pos_paper_field = train_pos_paper_field_all.copy()
        train_steps = min(num_pf, num_af)"""
        train_pos_idx_generator = IndexGenerator(batch_size=batch_size, num_data=train_steps)
        train_loss_list = []

        # train
        for itr in tqdm.tqdm(range(train_pos_idx_generator.num_iterations()),
                             desc=f'Epoch {epoch + 1:>3}/{epochs}'):
            # for itr in range(train_pos_idx_generator.num_iterations()):
            train_pos_idx_batch = train_pos_idx_generator.next()
            train_pos_idx_batch.sort()
            train_pos_af_batch = train_pos_author_field_all[train_pos_idx_batch].tolist()
            train_pos_pf_batch = train_pos_paper_field_all[train_pos_idx_batch].tolist()
            train_neg_idx_batch = np.random.choice(len(train_neg_author_field), len(train_pos_idx_batch))
            train_neg_idx_batch.sort()
            train_neg_af_batch = train_neg_author_field[train_neg_idx_batch].tolist()
            train_neg_pf_batch = train_neg_paper_field[train_neg_idx_batch].tolist()

            # af
            train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_lists, train_pos_af_batch, device,
                neighbor_samples, use_masks_af, offset=[num_paper, num_paper + num_author], af=True
            )
            train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_lists, train_neg_af_batch, device,
                neighbor_samples, no_masks, offset=[num_paper, num_paper + num_author], af=True
            )
            # pf
            train_pos_pf_g_lists, train_pos_pf_indices_lists, train_pos_pf_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_lists, train_pos_pf_batch, device, neighbor_samples,
                use_masks_pf, offset=[0, num_paper + num_author], af=False
            )
            train_neg_pf_g_lists, train_neg_pf_indices_lists, train_neg_pf_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_lists, train_neg_pf_batch, device, neighbor_samples,
                no_masks, offset=[0, num_paper + num_author], af=False
            )

            [pos_embedding_paper, pos_embedding_author, pos_embedding_field], _ = net((
                train_pos_g_lists, features_list, type_mask, train_pos_indices_lists,
                train_pos_idx_batch_mapped_lists))
            [neg_embedding_paper, neg_embedding_author, neg_embedding_field], _ = net((
                train_neg_g_lists, features_list, type_mask, train_neg_indices_lists,
                train_neg_idx_batch_mapped_lists))
            [pos_embedding_pf_paper, _, pos_embedding_pf_field], _ = net((
                train_pos_pf_g_lists, features_list, type_mask, train_pos_pf_indices_lists,
                train_pos_pf_idx_batch_mapped_lists))
            [neg_embedding_pf_paper, _, neg_embedding_pf_field], _ = net((
                train_neg_pf_g_lists, features_list, type_mask, train_neg_pf_indices_lists,
                train_neg_pf_idx_batch_mapped_lists))
            # af
            pos_embedding_author = pos_embedding_author.view(-1, 1, pos_embedding_author.shape[1])
            pos_embedding_field = pos_embedding_field.view(-1, pos_embedding_field.shape[1], 1)
            neg_embedding_author = neg_embedding_author.view(-1, 1, neg_embedding_author.shape[1])
            neg_embedding_field = neg_embedding_field.view(-1, neg_embedding_field.shape[1], 1)
            pos_af_out = torch.bmm(pos_embedding_author, pos_embedding_field)
            neg_af_out = -torch.bmm(neg_embedding_author, neg_embedding_field)
            # pf
            pos_embedding_pf_paper = pos_embedding_pf_paper.view(-1, 1, pos_embedding_pf_paper.shape[1])
            pos_embedding_pf_field = pos_embedding_pf_field.view(-1, pos_embedding_pf_field.shape[1], 1)
            neg_embedding_pf_paper = neg_embedding_pf_paper.view(-1, 1, neg_embedding_pf_paper.shape[1])
            neg_embedding_pf_field = neg_embedding_pf_field.view(-1, neg_embedding_pf_field.shape[1], 1)
            pos_pf_out = torch.bmm(pos_embedding_pf_paper, pos_embedding_pf_field)
            neg_pf_out = -torch.bmm(neg_embedding_pf_paper, neg_embedding_pf_field)

            train_loss = - torch.mean(F.logsigmoid(pos_af_out) + F.logsigmoid(neg_af_out)) \
                         - torch.mean(F.logsigmoid(pos_pf_out) + F.logsigmoid(neg_pf_out))
            if itr % 200 == 0:
                print(f"train loss in iter {itr} is: {train_loss.item()}")
            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # print(f'\rEpoch {epoch + 1:>3}/{epochs}: train loss = {train_loss.item():.6f}', end='')
            train_loss_list.append(train_loss.item())

        train_loss_avg = torch.mean(torch.tensor(train_loss_list))
        early_stopping(train_loss_avg, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--attn-vec-dim', type=int, default=128)
    ap.add_argument('--rnn-type', default='RotatE0')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--split', action='store_true')
    args = ap.parse_args()

    run(args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
        args.batch_size, args.epochs, args.patience, args.samples)
