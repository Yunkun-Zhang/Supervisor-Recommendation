from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
'''
def find_ap_path(author_field, field_paper):
    """
    find A-P pairs connected by either (1) a direct edge or (2) a AFP path.
    return: two lists containing AP/AFP nodes
    """
    # all AFP paths
    ap_dct = defaultdict(list)
    a_f_p = []
    a_f = []
    for a, f_list in tqdm.tqdm(author_field.items(), desc='edge A-F'):
        a_f.extend([(a, f) for f in f_list])
    for a, f in a_f:
        if not field_paper[f]:
            continue
        a_f_p.extend([(a, f, p) for p in field_paper[f]])
        ap_dct[a].append(p)
    a_f_p = np.array(a_f_p)
    # a_f_p[:, [0, 1]] += num_paper
    # a_f_p[:, 1] += num_author

    return a_f_p, ap_dct
'''


def get_train_val_test_split(target, val_split, test_split):
    train_target, mid_target = train_test_split(target, test_size=val_split + test_split)
    val_target, test_target = train_test_split(mid_target, test_size=test_split / (val_split + test_split))

    return train_target, val_target, test_target


def get_neg_samples(pos_dct, dim_2, sample_per_node):
    neg_samples = []
    for key, value in tqdm(pos_dct.items(), desc="Negative sampling"):
        candidate = np.setdiff1d(np.arange(dim_2), value)
        neg_target = np.random.choice(candidate, sample_per_node)
        for target in neg_target:
            neg_samples.append([key, target])
    return neg_samples

