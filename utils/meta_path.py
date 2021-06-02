"""
Definition and Acquirement of metapaths in sr-MAGNN

Node types:
P: Paper - 0
A: Author - 1
F: Fields - 2, c for child, p for parent

edges: A-P, A-F, P-F, Fc-Fp

Metapath types: FcFpFc, FpFcFp, APA, AFA, PAP, PFP, PFAFP, AFPFA
"""

from utils.process_data import *
import math
from utils.split_and_sample import *

expected_metapaths = [
    [(0, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 2, 0, 2, 1)],
    [(2, 2, 2)]
]


def find_meta_path(data, split=True):
    # index - id(in sql) mapping
    num_author, num_paper, num_field = len(data['authors']), len(data['papers']), len(data['fields'])
    dim = num_author + num_paper + num_field

    # create type mask
    type_mask = np.zeros(dim, dtype=int)
    type_mask[:num_paper] = 0  # 0 for paper
    type_mask[num_paper:num_paper + num_author] = 1  # 1 for author
    type_mask[num_paper + num_author:] = 2  # 2 for field
    np.save('../data/preprocessed/type_mask', type_mask)

    # dataset split
    author_field = defaultdict(list)
    for a, f in data['author_field']:
        author_field[a].append(f)
    paper_field = defaultdict(list)
    for p, f in data['paper_field']:
        paper_field[p].append(f)

    num = max(len(data['author_field']), len(data['paper_field']))
    neg_af = get_neg_samples(author_field, num_field, num // num_author + 1)
    neg_pf = get_neg_samples(paper_field, num_field, num // num_paper + 1)
    if split:
        af_pos_train, af_pos_val, af_pos_test = get_train_val_test_split(data['author_field'], 0.1, 0.1)
        pf_pos_train, pf_pos_val, pf_pos_test = get_train_val_test_split(data['paper_field'], 0.1, 0.1)
        af_neg_train, af_neg_val, af_neg_test = get_train_val_test_split(neg_af, 0.1, 0.1)
        pf_neg_train, pf_neg_val, pf_neg_test = get_train_val_test_split(neg_pf, 0.1, 0.1)
    else:
        af_pos_train, af_neg_train, pf_pos_train, pf_neg_train = data['author_field'], neg_af, data['paper_field'], neg_pf
        af_pos_val = af_pos_test = af_neg_val = af_neg_test = []
        pf_pos_val = pf_pos_test = pf_neg_val = pf_neg_test = []

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

    """af_pos = np.load('../data/preprocessed/train_val_test_pos_author_field.npz')
    pf_pos = np.load('../data/preprocessed/train_val_test_pos_paper_field.npz')
    af_pos_train = af_pos['train_pos_author_field']
    pf_pos_train = pf_pos['train_pos_paper_field']"""
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

    get_meta_paths(author_paper,
                   paper_author,
                   author_field,
                   field_author,
                   child_parent_field,
                   parent_child_field,
                   paper_field,
                   field_paper,
                   num_author,
                   num_paper,
                   num_field)


def get_meta_paths(author_paper, paper_author, author_field, field_author, child_parent_field,
                   parent_child_field, paper_field, field_paper, num_author, num_paper, num_field):
    target_idx_lists = [np.arange(num_paper), np.arange(num_author), np.arange(num_field)]
    offset_list = [0, num_paper, num_paper + num_author]

    def save(i, metapath, edge_metapath_idx_array):
        with open(f'../data/preprocessed/{i}/' + '-'.join(map(str, metapath)) + '_idx.pickle',
                  'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in tqdm(target_idx_lists[i], desc=f'Saving {metapath}_idx'):
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        with open(f'../data/preprocessed/{i}/' + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in tqdm(target_idx_lists[i], desc=f'Saving {metapath}.adjlist'):
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

    # create metapaths
    p_a_p = []
    for a, p_list in tqdm(author_paper.items(), desc='Metapath p-a-p'):
        p_a_p.extend([(p1, a, p2) for p1 in p_list for p2 in p_list])
    print('Sorting p-a-p...')
    p_a_p.sort(key=lambda x: [x[0], x[2], x[1]])
    p_a_p = np.array(p_a_p)
    p_a_p[:, 1] += num_paper
    save(0, (0, 1, 0), p_a_p)

    p_f_p = []
    for f, p_list in tqdm(field_paper.items(), desc='Metapath p-f-p'):
        fp = np.array(field_paper[f])
        p1_candidate_index = np.random.choice(len(fp), math.ceil(0.6 * len(fp)), replace=False)
        p2_candidate_index = np.random.choice(len(fp), math.ceil(0.6 * len(fp)), replace=False)
        p1_candidate = fp[p1_candidate_index]
        p2_candidate = fp[p2_candidate_index]
        p_f_p.extend([(p1, f, p2) for p1 in p1_candidate for p2 in p2_candidate])
    print('Sorting p-f-p...')
    p_f_p.sort(key=lambda x: [x[0], x[2], x[1]])
    p_f_p = np.array(p_f_p)
    p_f_p[:, 1] += num_paper + num_author
    save(0, (0, 2, 0), p_f_p)

    a_p_a = []
    for p, a_list in tqdm(paper_author.items(), desc='Metapath a-p-a'):
        a_p_a.extend([(a1, p, a2) for a1 in a_list for a2 in a_list])
    print('Sorting a-p-a...')
    a_p_a.sort(key=lambda x: [x[0], x[2], x[1]])
    a_p_a = np.array(a_p_a)
    a_p_a[:, [0, 2]] += num_paper
    save(1, (1, 0, 1), a_p_a)

    a_f_a = []
    for f, a_list in tqdm(field_author.items(), desc='Metapath a-f-a'):
        a_f_a.extend([(a1, f, a2) for a1 in a_list for a2 in a_list])
    print('Sorting a-f-a...')
    a_f_a.sort(key=lambda x: [x[0], x[2], x[1]])
    a_f_a = np.array(a_f_a)
    a_f_a += num_paper
    a_f_a[:, 1] += num_author
    save(1, (1, 2, 1), a_f_a)

    f_f_f = []  # both FcFpFc and FpFcFp
    for f, f_list in tqdm(parent_child_field.items(), desc='Metapath f-fp-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    for f, f_list in tqdm(child_parent_field.items(), desc='Metapath f-fc-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    print('Sorting f-f-f...')
    f_f_f.sort(key=lambda x: [x[0], x[2], x[1]])
    f_f_f = np.array(f_f_f)
    f_f_f += num_paper + num_author
    save(2, (2, 2, 2), f_f_f)

    p_f_a_f_p = []
    # first obtain FAF
    f_a_f = []
    for a, f_list in tqdm(author_field.items(), desc="Metapath f-a-f"):
        f_a_f.extend([(f1, a, f2) for f1 in f_list for f2 in f_list])
    # f_a_f = np.array(f_a_f)
    # f_a_f += num_paper
    # f_a_f[:, [0, 2]] += num_author
    # next extends to two sides for PFAFP
    for f1, a, f2 in tqdm(f_a_f, desc='Metapath p-f-a-f-p'):
        # check if these fields are connected to any author
        if not field_paper[f1] or not field_paper[f2]:
            continue
        fp1 = np.array(field_paper[f1])
        fp2 = np.array(field_paper[f2])
        p1_candidate_index = np.random.choice(len(fp1), math.ceil(0.2 * len(fp1)), replace=False)
        p1_candidate = fp1[p1_candidate_index]
        p2_candidate_index = np.random.choice(len(fp2), math.ceil(0.2 * len(fp2)), replace=False)
        p2_candidate = fp2[p2_candidate_index]
        p_f_a_f_p.extend([(p1, f1, a, f2, p2) for p1 in p1_candidate for p2 in p2_candidate])
    print('Sorting p-f-a-f-p...')
    p_f_a_f_p.sort(key=lambda x: [x[0], x[4], x[1], x[2], x[3]])
    p_f_a_f_p = np.array(p_f_a_f_p)
    p_f_a_f_p[:, [1, 2, 3]] += num_paper
    p_f_a_f_p[:, [1, 3]] += num_author
    save(0, (0, 2, 1, 2, 0), p_f_a_f_p)

    a_f_p_f_a = []
    # first obtain f_p_f
    f_p_f = []
    for p, f_list in tqdm(paper_field.items(), desc="Metapath f-p-f"):
        f_p_f.extend([(f1, p, f2) for f1 in f_list for f2 in f_list])
    # f_p_f = np.array(f_p_f)
    for f1, p, f2 in tqdm(f_p_f, desc='Metapath a-f-p-f-a'):
        # check if these fields are connected to any author
        if not field_author[f1] or not field_author[f2]:
            continue
        fa1 = np.array(field_author[f1])
        fa2 = np.array(field_author[f2])
        a1_candidate_index = np.random.choice(len(fa1), math.ceil(0.2 * len(fa1)), replace=False)
        a1_candidate = fa1[a1_candidate_index]
        a2_candidate_index = np.random.choice(len(fa2), math.ceil(0.2 * len(fa2)), replace=False)
        a2_candidate = fa2[a2_candidate_index]
        a_f_p_f_a.extend([(a1, f1, p, f2, a2) for a1 in a1_candidate for a2 in a2_candidate])
    print('Sorting a-f-p-f-a...')
    a_f_p_f_a.sort(key=lambda x: [x[0], x[4], x[1], x[2], x[3]])
    a_f_p_f_a = np.array(a_f_p_f_a)
    a_f_p_f_a[:, [0, 1, 3, 4]] += num_paper
    a_f_p_f_a[:, [1, 3]] += num_author
    save(1, (1, 2, 0, 2, 1), a_f_p_f_a)


def load_mp(path='../data/preprocessed'):
    adj_lists, idx_lists = [], []
    for mode in range(3):
        adj_list, idx_list = [], []
        for metapath in expected_metapaths[mode]:
            with open(path + f'/{mode}/' + '-'.join(map(str, metapath)) + '.adjlist', 'r') as f:
                adj_list.append([line.strip() for line in f])
            with open(path + f'/{mode}/' + '-'.join(map(str, metapath)) + '_idx.pickle', 'rb') as f:
                idx_list.append(pickle.load(f))
        adj_lists.append(adj_list)
        idx_lists.append(idx_list)
    type_mask = np.load(path + '/type_mask.npy')
    pos_af = np.load(path + '/train_val_test_pos_author_field.npz')
    pos_pf = np.load(path + '/train_val_test_pos_paper_field.npz')
    neg_af = np.load(path + '/train_val_test_neg_author_field.npz')
    neg_pf = np.load(path + '/train_val_test_neg_paper_field.npz')
    return adj_lists, idx_lists, type_mask, pos_af, neg_af, pos_pf, neg_pf


if __name__ == '__main__':
    data = load_data('../data/CS+med')
    find_meta_path(data, split=True)
