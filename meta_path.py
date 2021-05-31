from process_data import *

"""
0 for paper
1 for author
2 for field   
"""


def find_meta_path(data):
    num_author, num_paper, num_field = len(data['authors']), len(data['papers']), len(data['fields'])
    dim = num_author + num_paper + num_field

    # create type mask
    type_mask = np.zeros(dim, dtype=int)
    type_mask[:num_paper] = 0  # 0 for paper
    type_mask[num_paper:num_paper + num_author] = 1  # 1 for author
    type_mask[num_paper + num_author:] = 2  # 2 for field

    # create dicts for metapath
    author_paper = defaultdict(list)
    paper_author = defaultdict(list)
    for a, p in data['author_paper']:
        author_paper[a].append(p)
        paper_author[p].append(a)
    author_field = defaultdict(list)
    field_author = defaultdict(list)
    for a, f in data['author_field']:
        author_field[a].append(f)
        field_author[f].append(a)
    child_parent_field = defaultdict(list)
    parent_child_field = defaultdict(list)
    for c, p in data['field_parent']:
        child_parent_field[c].append(p)
        parent_child_field[p].append(c)
    paper_field = defaultdict(list)
    field_paper = defaultdict(list)
    for p, f in data['paper_field']:
        paper_field[p].append(f)
        field_paper[f].append(p)

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

    f_f_f = []
    for f, f_list in tqdm.tqdm(parent_child_field.items(), desc='Meta-path f-fp-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    for f, f_list in tqdm.tqdm(child_parent_field.items(), desc='Meta-path f-fc-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    f_f_f = np.array(f_f_f)
    f_f_f += num_paper + num_author

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
