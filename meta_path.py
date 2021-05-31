"""
Definition and Acquirement of metapaths in sr-MAGNN

Node types:
P: Paper - 0
A: Author - 1
F: Fields - 2, c for child, p for parent

edges: A-P, A-F, P-F, Fc-Fp

Metapath types: FcFpFc, FpFcFp, APA, AFA, PAP, PFP, PFAFP, AFPFA
"""
import numpy as np
from process_data import *
import math
from split_and_sample import *

def find_meta_path(data):
    num_author, num_paper, num_field = len(data['authors']), len(data['papers']), len(data['fields'])
    dim = num_author + num_paper + num_field

    # create type mask
    type_mask = np.zeros(dim, dtype=int)
    type_mask[:num_paper] = 0  # 0 for paper
    type_mask[num_paper:num_paper + num_author] = 1  # 1 for author
    type_mask[num_paper + num_author:] = 2  # 2 for field

    # dataset split
    neg_af = get_neg_samples(data['author_field'], num_author, num_field)
    neg_pf = get_neg_samples(data['paper_field'], num_paper, num_field)
    af_pos_train, af_pos_val, af_pos_test = get_train_val_test_split(data['author_field'], 0.1, 0.1)
    pf_pos_train, pf_pos_val, pf_pos_test = get_train_val_test_split(data['paper_field'], 0.1, 0.1)
    af_neg_train, af_neg_val, af_neg_test = get_train_val_test_split(neg_af, 0.1, 0.1)
    pf_neg_train, pf_neg_val, pf_neg_test = get_train_val_test_split(neg_pf, 0.1, 0.1)

    np.savez('../data/preprocessed/train_val_test_neg_author_field.npz',
             train_neg_author_field=af_neg_train,
             val_neg_author_field=af_neg_val,
             test_neg_author_field=af_neg_test)
    np.savez('../data/preprocessed/train_val_test_pos_author_field.npz',
             train_pos_author_field=af_pos_train,
             val_pos_author_field=af_pos_val,
             test_pos_author_field=af_pos_test)
    np.savez('../data/preprocessed/train_val_test_neg_paper_field.npz',
             train_neg_paper_field=pf_neg_train,
             val_neg_paper_field=pf_neg_val,
             test_neg_paper_field=pf_neg_test)
    np.savez('../data/preprocessed/train_val_test_pos_paper_field.npz',
             train_pos_paper_field=pf_pos_train,
             val_pos_paper_field=pf_pos_val,
             test_pos_paper_field=pf_pos_test)



    # create dicts for metapath
    # each dict has form: key: source - value: targets
    author_paper = defaultdict(list)
    paper_author = defaultdict(list)
    for a, p in data['author_paper']:
        author_paper[a].append(p)
        paper_author[p].append(a)
    author_field = defaultdict(list)
    field_author = defaultdict(list)
    for a, f in af_pos_train:
        author_field[a].append(f)
        field_author[f].append(a)
    child_parent_field = defaultdict(list)
    parent_child_field = defaultdict(list)
    for c, p in data['field_parent']:
        child_parent_field[c].append(p)
        parent_child_field[p].append(c)
    paper_field = defaultdict(list)
    field_paper = defaultdict(list)
    for p, f in pf_pos_train:
        paper_field[p].append(f)
        field_paper[f].append(p)


    p_a_p, p_f_p, a_p_a, a_f_a, f_f_f, p_f_a_f_p, a_f_p_f_a = get_meta_paths(author_paper,
                                                                             paper_author,
                                                                             author_field,
                                                                             field_author,
                                                                             child_parent_field,
                                                                             parent_child_field,
                                                                             paper_field,
                                                                             field_paper,
                                                                             num_author,
                                                                             num_paper)

    # save metapaths
    expected_metapaths = [
        [(0, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
        [(1, 0, 1), (1, 2, 1), (1, 2, 0, 2, 1)],
        [(2, 2, 2)]
    ]
    metapath_indices_mapping = {(0, 1, 0): p_a_p, (1, 2, 1): a_f_a}

    target_idx_lists = [np.arange(num_paper), np.arange(num_author)]
    offset_list = [0, num_paper]
    for i, metapaths in enumerate(expected_metapaths):
        for metapath in metapaths:
            edge_metapath_idx_array = metapath_indices_mapping[metapath]

            with open(f'data/preprocessed/{i}/' + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
                target_metapaths_mapping = {}
                left = 0
                right = 0
                for target_idx in tqdm.tqdm(target_idx_lists[i], desc=f'Saving {metapath}_idx'):
                    while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                            offset_list[i]:
                        right += 1
                    target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                    left = right
                pickle.dump(target_metapaths_mapping, out_file)

            with open(f'data/preprocessed/{i}/' + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
                left = 0
                right = 0
                for target_idx in tqdm.tqdm(target_idx_lists[i], desc=f'Saving {metapath}.adjlist'):
                    while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                            offset_list[i]:
                        right += 1
                    neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                    neighbors = list(map(str, neighbors))
                    if len(neighbors) > 0:
                        out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                    else:
                        out_file.write('{}\n'.format(target_idx))
                    left = right


def get_meta_paths(author_paper, paper_author, author_field, field_author,
                   child_parent_field, parent_child_field, paper_field, field_paper, num_author, num_paper):
    # create metapaths
    p_a_p = []
    for a, p_list in tqdm.tqdm(author_paper.items(), desc='Meta-path p-a-p'):
        p_a_p.extend([(p1, a, p2) for p1 in p_list for p2 in p_list])
    p_a_p = np.array(p_a_p)
    p_a_p[:, 1] += num_paper

    p_f_p = []
    for f, p_list in tqdm.tqdm(field_paper.items(), desc='Meta-path p-f-f'):
        p_f_p.extend([(p1, f, p2) for p1 in p_list for p2 in p_list])
    p_f_p = np.array(p_f_p)
    p_f_p[:, 1] += num_paper + num_author

    a_p_a = []
    for p, a_list in tqdm.tqdm(paper_author.items(), desc='Meta-path a-p-a'):
        a_p_a.extend([(a1, p, a2) for a1 in a_list for a2 in a_list])
    a_p_a = np.array(a_p_a)
    a_p_a[:, [0, 2]] += num_paper

    a_f_a = []
    for f, a_list in tqdm.tqdm(field_author.items(), desc='Meta-path a-f-a'):
        a_f_a.extend([(a1, f, a2) for a1 in a_list for a2 in a_list])
    a_f_a = np.array(a_f_a)
    a_f_a += num_paper
    a_f_a[:, 1] += num_author

    f_f_f = []  # both FcFpFc and FpFcFp
    for f, f_list in tqdm.tqdm(parent_child_field.items(), desc='Meta-path f-fp-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    for f, f_list in tqdm.tqdm(child_parent_field.items(), desc='Meta-path f-fc-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    f_f_f = np.array(f_f_f)
    f_f_f += num_paper + num_author

    p_f_a_f_p = []
    # first obtain FAF
    f_a_f = []
    for a, f_list in tqdm.tqdm(author_field.items(), des="Meta-path f-a-f"):
        f_a_f.extend([(f1, a, f2) for f1 in f_list for f2 in f_list])
    # f_a_f = np.array(f_a_f)
    # f_a_f += num_paper
    # f_a_f[:, [0, 2]] += num_author
    # next extends to two sides for PFAFP
    for f1, a, f2 in f_a_f:
        # check if these fields are connected to any author
        if not field_paper[f1] or not field_paper[f2]:
            continue
        p1_candidate_index = np.random.choice(len(field_paper[f1]), math.ceil(0.2 * len(field_paper[f1])),
                                              replace=False)
        p1_candidate = field_paper[f1][p1_candidate_index]
        p2_candidate_index = np.random.choice(len(field_paper[f2]), math.ceil(0.2 * len(field_paper[f2])),
                                              replace=False)
        p2_candidate = field_paper[f2][p2_candidate_index]
        p_f_a_f_p.extend([(p1, f1, a, f2, p2) for p1 in p1_candidate for p2 in p2_candidate])
    p_f_a_f_p = np.array(p_f_a_f_p)
    p_f_a_f_p[:, [1, 2, 3]] += num_paper
    p_f_a_f_p[:, [1, 3]] += num_author

    a_f_p_f_a = []
    # first obtain f_p_f
    f_p_f = []
    for p, f_list in tqdm.tqdm(paper_field.items(), des="Meta-path f-a-f"):
        f_p_f.extend([(f1, p, f2) for f1 in f_list for f2 in f_list])
    # f_p_f = np.array(f_p_f)
    for f1, p, f2 in f_p_f:
        # check if these fields are connected to any author
        if not field_author[f1] or not field_author[f2]:
            continue
        a1_candidate_index = np.random.choice(len(field_author[f1]), math.ceil(0.2 * len(field_author[f1])),
                                              replace=False)
        a1_candidate = field_author[f1][a1_candidate_index]
        a2_candidate_index = np.random.choice(len(field_author[f2]), math.ceil(0.2 * len(field_author[f2])),
                                              replace=False)
        a2_candidate = field_author[f2][a2_candidate_index]
        a_f_p_f_a.extend([(a1, f1, p, f2, a2) for a1 in a1_candidate for a2 in a2_candidate])
    a_f_p_f_a = np.array(a_f_p_f_a)
    a_f_p_f_a[:, [0, 1, 3, 4]] += num_paper
    a_f_p_f_a[:, [1, 3]] += num_author

    return p_a_p, p_f_p, a_p_a, a_f_a, f_f_f, p_f_a_f_p, a_f_p_f_a
