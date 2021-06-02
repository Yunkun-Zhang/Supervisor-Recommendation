import pickle
import torch
import numpy as np
from models import LP
from utils.tools import parse_test
from utils.meta_path import load_mp
from utils.process_data import list_to_dict

num_paper = 44016
num_author = 16571
hidden_dim = 64
num_heads = 8
attn_vec_dim = 128
rnn_type = 'RotatE0'
etype_lists = [[[0, 1], [2, 3], [2, 5, 4, 3]],
               [[1, 0], [4, 5], [4, 3, 2, 5]],
               [[None, None]]]


def predict_and_save_emb():
    adjlists, edge_metapath_indices_lists, type_mask, _, _, _, _ = load_mp('data/preprocessed')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    # feats_type = 1
    for i in range(3):
        dim = 10
        num_nodes = (type_mask == i).sum()
        in_dims.append(dim)
        features_list.append(torch.zeros((num_nodes, 10)).to(device))

    net = LP([3, 3, 1], 6, etype_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, 0.5)
    net.to(device)
    net.load_state_dict(torch.load('checkpoint/checkpoint.pt'))
    net.eval()
    with torch.no_grad():
        paper_embs = []
        author_embs = []
        left = 0
        while left < num_paper:
            right = min(left + 128, num_paper)
            batch = list(range(left, right))
            g_list_paper, indices_list_paper, mapped_list_paper = parse_test(
                adjlists, edge_metapath_indices_lists, batch, device, 0)
            [paper_emb, _, _], _ = net(([g_list_paper, [], []],
                                        features_list,
                                        type_mask,
                                        [indices_list_paper, [], []],
                                        [mapped_list_paper, [], []]))
            paper_embs.append(paper_emb.to('cpu').numpy())
            left = right
        paper_embs = np.concatenate(paper_embs, axis=0)
        left = 0
        while left < num_author:
            right = min(left + 128, num_author)
            batch = list(range(left, right))
            g_list_author, indices_list_author, mapped_list_author = parse_test(
                adjlists, edge_metapath_indices_lists, batch, device, 1)
            [_, author_emb, _], _ = net(([[], g_list_author, []],
                                         features_list,
                                         type_mask,
                                         [[], indices_list_author, []],
                                         [[], mapped_list_author, []]))
            author_embs.append(author_emb.to('cpu').numpy())
            left = right
        author_embs = np.concatenate(author_embs, axis=0)
        print(paper_embs.shape)
        print(author_embs.shape)
        # np.save('embeddings/paper_embs', paper_embs)
        # np.save('embeddings/author_embs', author_embs)


def predict(paper_id_list=None, author_id_list=None):
    adjlists, edge_metapath_indices_lists, type_mask, _, _, _, _ = load_mp('data/preprocessed')
    device = torch.device('cpu')
    features_list = []
    in_dims = []
    # feats_type = 1
    for i in range(3):
        dim = 10
        num_nodes = (type_mask == i).sum()
        in_dims.append(dim)
        features_list.append(torch.zeros((num_nodes, 10)))

    net = LP([3, 3, 1], 6, etype_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, 0.5)
    net.load_state_dict(torch.load('checkpoint/checkpoint.pt'))
    net.eval()

    with torch.no_grad():
        paper_emb = author_emb = None
        if paper_id_list is not None:
            papers = np.load('embeddings/paper_embs.npy')
            paper_dict = list_to_dict(papers)
            paper_list = [paper_dict[paper_id] for paper_id in paper_id_list]
            g_list_paper, indices_list_paper, mapped_list_paper = parse_test(
                adjlists, edge_metapath_indices_lists, paper_list, device, 0)
            [paper_emb, _, _], _ = net(([g_list_paper, [], []],
                                        features_list,
                                        type_mask,
                                        [indices_list_paper, [], []],
                                        [mapped_list_paper, [], []]))
        if author_id_list is not None:
            with open('data/CS+med/filtered_authors.pickle', 'rb') as f:
                authors = pickle.load(f)
            author_dict = list_to_dict([a['id'] for a in authors])
            author_list = [author_dict[author_id] for author_id in author_id_list]
            g_list_author, indices_list_author, mapped_list_author = parse_test(
                adjlists, edge_metapath_indices_lists, author_list, device, 1)
            [_, author_emb, _], _ = net(([[], g_list_author, []],
                                         features_list,
                                         type_mask,
                                         [[], indices_list_author, []],
                                         [[], mapped_list_author, []]))
    return paper_emb, author_emb


if __name__ == '__main__':
    predict_and_save_emb()
