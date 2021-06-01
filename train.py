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
               [[6, 6]]]
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
    adjlists, edge_metapath_indices_list, type_mask, pos_af, neg_af, pos_pf, neg_pf = load_mp('data/preprocessed')
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
    val_pos_author_field_all = pos_af['val_pos_author_field']
    val_pos_paper_field_all = pos_pf['val_pos_paper_field']
    test_pos_author_field = pos_af['test_pos_author_field']
    test_pos_paper_field = pos_pf['test_pos_paper_field']
    train_neg_author_field = neg_af['train_neg_author_field']
    train_neg_paper_field = neg_pf['train_neg_paper_field']
    val_neg_author_field = neg_af['val_neg_author_field']
    val_neg_paper_field = neg_pf['val_neg_paper_field']
    test_neg_author_field = neg_af['test_neg_author_field']
    test_neg_paper_field = neg_pf['test_neg_paper_field']

    y_true_af = np.array([1] * len(test_pos_author_field) + [0] * len(test_neg_author_field))
    y_true_pf = np.array([1] * len(test_pos_paper_field) + [0] * len(test_neg_paper_field))
    # auc_af, auc_pf, ap_af, ap_pf = [], [], [], []

    # define model
    net = LP([3, 3, 1], 7, etype_lists, in_dims, hidden_dim, hidden_dim,
             num_heads, attn_vec_dim, rnn_type, dropout_rate)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   save_path=f'checkpoint/checkpoint.pt')

    # run
    for epoch in range(epochs):
        # adjust sample counts
        num_af, num_pf = len(train_pos_author_field_all), len(train_pos_paper_field_all)
        if num_af < num_pf:
            train_pos_paper_field = train_pos_paper_field_all[np.random.choice(num_pf, num_af, replace=False)]
            train_pos_author_field = train_pos_author_field_all.copy()
            # train_neg_paper_field = train_neg_paper_field[np.random.choice(num_pf, num_af, replace=False)]
        else:
            train_pos_author_field = train_pos_author_field_all[np.random.choice(num_af, num_pf, replace=False)]
            train_pos_paper_field = train_pos_paper_field_all.copy()
            # train_neg_author_field = np.random.choice(train_neg_author_field, num_pf, replace=False)
        train_steps = min(num_pf, num_af)
        train_pos_idx_generator = IndexGenerator(batch_size=batch_size, num_data=train_steps)

        num_af, num_pf = len(val_pos_author_field_all), len(val_pos_paper_field_all)
        if num_af < num_pf:
            val_pos_paper_field = val_pos_paper_field_all[np.random.choice(num_pf, num_af, replace=False)]
            val_pos_author_field = val_pos_author_field_all.copy()
            # val_neg_paper_field = np.random.choice(val_neg_paper_field, num_af, replace=False)
        else:
            val_pos_author_field = val_pos_author_field_all[np.random.choice(num_af, num_pf, replace=False)]
            val_pos_paper_field = val_pos_paper_field_all.copy()
            # val_neg_author_field = np.random.choice(val_neg_author_field, num_pf, replace=False)
        val_steps = min(num_af, num_pf)
        val_idx_generator = IndexGenerator(batch_size=batch_size, num_data=val_steps, shuffle=False)

        # train
        for itr in tqdm.tqdm(range(train_pos_idx_generator.num_iterations()),
                             desc=f'Epoch {epoch + 1:>3}/{epochs}'):
            train_pos_idx_batch = train_pos_idx_generator.next()
            train_pos_idx_batch.sort()
            train_pos_af_batch = train_pos_author_field[train_pos_idx_batch].tolist()
            train_pos_pf_batch = train_pos_paper_field[train_pos_idx_batch].tolist()
            train_neg_idx_batch = np.random.choice(len(train_neg_author_field), len(train_pos_idx_batch))
            train_neg_idx_batch.sort()
            train_neg_af_batch = train_neg_author_field[train_neg_idx_batch].tolist()
            train_neg_pf_batch = train_neg_paper_field[train_neg_idx_batch].tolist()

            train_pos_af_g_lists, train_pos_af_indices_lists, train_pos_af_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_pos_af_batch, device, neighbor_samples,
                use_masks_af, offset=[num_paper, num_paper + num_author]
            )
            train_pos_pf_g_lists, train_pos_pf_indices_lists, train_pos_pf_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_pos_pf_batch, device, neighbor_samples,
                use_masks_pf, offset=[0, num_paper + num_author]
            )
            train_neg_af_g_lists, train_neg_af_indices_lists, train_neg_af_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_neg_af_batch, device, neighbor_samples,
                no_masks, offset=[num_paper, num_paper + num_author]
            )
            train_neg_pf_g_lists, train_neg_pf_indices_lists, train_neg_pf_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_neg_pf_batch, device, neighbor_samples,
                no_masks, offset=[0, num_paper + num_author]
            )

            [_, pos_embedding_af_author, pos_embedding_af_field], _ = net((
                train_pos_af_g_lists, features_list, type_mask, train_pos_af_indices_lists,
                train_pos_af_idx_batch_mapped_lists))
            [_, neg_embedding_af_author, neg_embedding_af_field], _ = net((
                train_neg_af_g_lists, features_list, type_mask, train_neg_af_indices_lists,
                train_neg_af_idx_batch_mapped_lists))
            [pos_embedding_pf_paper, _, pos_embedding_pf_field], _ = net((
                train_pos_pf_g_lists, features_list, type_mask, train_pos_pf_indices_lists,
                train_pos_pf_idx_batch_mapped_lists))
            [neg_embedding_pf_paper, _, neg_embedding_pf_field], _ = net((
                train_neg_pf_g_lists, features_list, type_mask, train_neg_pf_indices_lists,
                train_neg_pf_idx_batch_mapped_lists))
            pos_embedding_af_author = pos_embedding_af_author.view(-1, 1, pos_embedding_af_author.shape[1])
            pos_embedding_af_field = pos_embedding_af_field.view(-1, pos_embedding_af_field.shape[1], 1)
            neg_embedding_af_author = neg_embedding_af_author.view(-1, 1, neg_embedding_af_author.shape[1])
            neg_embedding_af_field = neg_embedding_af_field.view(-1, neg_embedding_af_field.shape[1], 1)
            pos_af_out = torch.bmm(pos_embedding_af_author, pos_embedding_af_field)
            neg_af_out = -torch.bmm(neg_embedding_af_author, neg_embedding_af_field)
            pos_embedding_pf_paper = pos_embedding_pf_paper.view(-1, 1, pos_embedding_pf_paper.shape[1])
            pos_embedding_pf_field = pos_embedding_pf_field.view(-1, pos_embedding_pf_field.shape[1], 1)
            neg_embedding_pf_paper = neg_embedding_pf_paper.view(-1, 1, neg_embedding_pf_paper.shape[1])
            neg_embedding_pf_field = neg_embedding_pf_field.view(-1, neg_embedding_pf_field.shape[1], 1)
            pos_pf_out = torch.bmm(pos_embedding_pf_paper, pos_embedding_pf_field)
            neg_pf_out = -torch.bmm(neg_embedding_pf_paper, neg_embedding_pf_field)

            train_loss = - torch.mean(F.logsigmoid(pos_af_out) + F.logsigmoid(neg_af_out)) \
                         - torch.mean(F.logsigmoid(pos_pf_out) + F.logsigmoid(neg_pf_out))

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # validation
        net.eval()
        val_loss = []
        with torch.no_grad():
            for itr in range(val_idx_generator.num_iterations()):
                val_idx_batch = val_idx_generator.next()
                val_pos_af_batch = val_pos_author_field[val_idx_batch].tolist()
                val_pos_pf_batch = val_pos_paper_field[val_idx_batch].tolist()
                val_neg_af_batch = val_neg_author_field[val_idx_batch].tolist()
                val_neg_pf_batch = val_neg_paper_field[val_idx_batch].tolist()
                val_pos_af_g_lists, val_pos_af_indices_lists, val_pos_af_idx_batch_mapped_lists = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_pos_af_batch, device, neighbor_samples,
                    no_masks, [num_paper, num_paper + num_author])
                val_pos_pf_g_lists, val_pos_pf_indices_lists, val_pos_pf_idx_batch_mapped_lists = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_pos_pf_batch, device, neighbor_samples,
                    no_masks, [0, num_paper + num_author])
                val_neg_af_g_lists, val_neg_af_indices_lists, val_neg_af_idx_batch_mapped_lists = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_neg_af_batch, device, neighbor_samples,
                    no_masks, [num_paper, num_paper + num_author])
                val_neg_pf_g_lists, val_neg_pf_indices_lists, val_neg_pf_idx_batch_mapped_lists = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_neg_pf_batch, device, neighbor_samples,
                    no_masks, [0, num_paper + num_author])

                [_, pos_embedding_af_author, pos_embedding_af_field], _ = net((
                    val_pos_af_g_lists, features_list, type_mask, val_pos_af_indices_lists,
                    val_pos_af_idx_batch_mapped_lists))
                [_, neg_embedding_af_author, neg_embedding_af_field], _ = net((
                    val_neg_af_g_lists, features_list, type_mask, val_neg_af_indices_lists,
                    val_neg_af_idx_batch_mapped_lists))
                [pos_embedding_pf_paper, _, pos_embedding_pf_field], _ = net((
                    val_pos_pf_g_lists, features_list, type_mask, val_pos_pf_indices_lists,
                    val_pos_pf_idx_batch_mapped_lists))
                [neg_embedding_pf_paper, _, neg_embedding_pf_field], _ = net((
                    val_neg_pf_g_lists, features_list, type_mask, val_neg_pf_indices_lists,
                    val_neg_pf_idx_batch_mapped_lists))
                pos_embedding_af_author = pos_embedding_af_author.view(-1, 1, pos_embedding_af_author.shape[1])
                pos_embedding_af_field = pos_embedding_af_field.view(-1, pos_embedding_af_field.shape[1], 1)
                neg_embedding_af_author = neg_embedding_af_author.view(-1, 1, neg_embedding_af_author.shape[1])
                neg_embedding_af_field = neg_embedding_af_field.view(-1, neg_embedding_af_field.shape[1], 1)
                pos_af_out = torch.bmm(pos_embedding_af_author, pos_embedding_af_field)
                neg_af_out = -torch.bmm(neg_embedding_af_author, neg_embedding_af_field)
                pos_embedding_pf_paper = pos_embedding_pf_paper.view(-1, 1, pos_embedding_pf_paper.shape[1])
                pos_embedding_pf_field = pos_embedding_pf_field.view(-1, pos_embedding_pf_field.shape[1], 1)
                neg_embedding_pf_paper = neg_embedding_pf_paper.view(-1, 1, neg_embedding_pf_paper.shape[1])
                neg_embedding_pf_field = neg_embedding_pf_field.view(-1, neg_embedding_pf_field.shape[1], 1)
                pos_pf_out = torch.bmm(pos_embedding_pf_paper, pos_embedding_pf_field)
                neg_pf_out = -torch.bmm(neg_embedding_pf_paper, neg_embedding_pf_field)

                val_loss.append(- torch.mean(F.logsigmoid(pos_af_out) + F.logsigmoid(neg_af_out))
                                - torch.mean(F.logsigmoid(pos_pf_out) + F.logsigmoid(neg_pf_out)))
            val_loss = torch.mean(torch.tensor(val_loss))
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

    # test AF
    net.load_state_dict(torch.load('checkpoint/checkpoint.pt'))
    test_af_idx_generator = IndexGenerator(batch_size=batch_size, num_data=len(test_pos_author_field), shuffle=False)
    net.eval()
    pos_proba_list = []
    neg_proba_list = []
    with torch.no_grad():
        for itr in range(test_af_idx_generator.num_iterations()):
            test_idx_batch = test_af_idx_generator.next()
            test_pos_batch = test_pos_author_field[test_idx_batch].tolist()
            test_neg_batch = test_neg_author_field[test_idx_batch].tolist()
            test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, test_pos_batch, device, neighbor_samples, no_masks,
                [num_paper, num_paper + num_author])
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, test_neg_batch, device, neighbor_samples, no_masks,
                [num_paper, num_paper + num_author])

            [_, pos_embedding_author, pos_embedding_field], _ = net(
                (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
            [_, neg_embedding_author, neg_embedding_field], _ = net(
                (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
            pos_embedding_author = pos_embedding_author.view(-1, 1, pos_embedding_author.shape[1])
            pos_embedding_field = pos_embedding_field.view(-1, pos_embedding_field.shape[1], 1)
            neg_embedding_author = neg_embedding_author.view(-1, 1, neg_embedding_author.shape[1])
            neg_embedding_field = neg_embedding_field.view(-1, neg_embedding_field.shape[1], 1)

            pos_out = torch.flatten(torch.bmm(pos_embedding_author, pos_embedding_field))
            neg_out = torch.flatten(torch.bmm(neg_embedding_author, neg_embedding_field))
            pos_proba_list.append(torch.sigmoid(pos_out))
            neg_proba_list.append(torch.sigmoid(neg_out))
        y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
        y_proba_test = y_proba_test.cpu().numpy()
    auc = roc_auc_score(y_true_af, y_proba_test)
    ap = average_precision_score(y_true_af, y_proba_test)
    print('Test Author-Field:')
    print(f'AUC = {auc}')
    print(f'AP = {ap}')

    # test PF
    test_pf_idx_generator = IndexGenerator(batch_size=batch_size, num_data=len(test_pos_paper_field), shuffle=False)
    net.eval()
    pos_proba_list = []
    neg_proba_list = []
    with torch.no_grad():
        for itr in range(test_pf_idx_generator.num_iterations()):
            test_idx_batch = test_pf_idx_generator.next()
            test_pos_batch = test_pos_paper_field[test_idx_batch].tolist()
            test_neg_batch = test_neg_paper_field[test_idx_batch].tolist()
            test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, test_pos_batch, device, neighbor_samples, no_masks,
                [0, num_paper + num_author])
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch(
                adjlists, edge_metapath_indices_list, test_neg_batch, device, neighbor_samples, no_masks,
                [0, num_paper + num_author])

            [_, pos_embedding_author, pos_embedding_field], _ = net(
                (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
            [_, neg_embedding_author, neg_embedding_field], _ = net(
                (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
            pos_embedding_author = pos_embedding_author.view(-1, 1, pos_embedding_author.shape[1])
            pos_embedding_field = pos_embedding_field.view(-1, pos_embedding_field.shape[1], 1)
            neg_embedding_author = neg_embedding_author.view(-1, 1, neg_embedding_author.shape[1])
            neg_embedding_field = neg_embedding_field.view(-1, neg_embedding_field.shape[1], 1)

            pos_out = torch.flatten(torch.bmm(pos_embedding_author, pos_embedding_field))
            neg_out = torch.flatten(torch.bmm(neg_embedding_author, neg_embedding_field))
            pos_proba_list.append(torch.sigmoid(pos_out))
            neg_proba_list.append(torch.sigmoid(neg_out))
        y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
        y_proba_test = y_proba_test.cpu().numpy()
    auc = roc_auc_score(y_true_pf, y_proba_test)
    ap = average_precision_score(y_true_pf, y_proba_test)
    print('Test Paper-Field:')
    print(f'AUC = {auc}')
    print(f'AP = {ap}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--attn-vec-dim', type=int, default=128)
    ap.add_argument('--rnn-type', default='RotatE0')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--samples', type=int, default=100)
    args = ap.parse_args()

    run(args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
        args.batch_size, args.epochs, args.patience, args.samples)
