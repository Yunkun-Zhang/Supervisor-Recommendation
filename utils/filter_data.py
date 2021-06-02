import pymysql
import numpy as np
import pickle
from utils.process_data import list_to_dict, defaultdict

old_path = '../data/old2/'


def get_selected_fields():
    """
    filter by level 0 field = Computer Science
    """
    all_selected_fields = set()
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_paper'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()

    target_id = ['2030591755']
    for _ in range(3):
        query = 'select field_id from am_paper.am_field_relation where parent_id'
        if len(target_id) == 1:
            query += f'={target_id[0]}'
        else:
            query += f' in{tuple(target_id)}'
        cursor.execute(query)
        res = cursor.fetchall()
        target_id = []
        for field in res:
            target_id.append(field[0])
            all_selected_fields.add(field[0])

    return all_selected_fields


def filter_other(old_fields, fields, name):
    other_field = np.load(old_path + f'{name}_field.npy')
    field_other = defaultdict(list)  # field_id: other_old_index
    for o, f in other_field:
        if old_fields[f] in fields:
            field_other[old_fields[f]].append(o)
    count = 0
    new_other_dict = {}  # old_index: new_index
    for fid, o_list in field_other.items():
        for o in o_list:
            if o not in new_other_dict:
                new_other_dict[o] = count
                count += 1
    return field_other, new_other_dict


def create_other_field_edges(field_other, new_other_dict, field_dict):
    new_other_field = []
    for f, o_list in field_other.items():
        for o in o_list:
            new_other_field.append([new_other_dict[o], field_dict[f]])
    return new_other_field


def create_others(new_other_dict, others):
    new_others = [-1 for _ in range(len(new_other_dict))]
    for o in new_other_dict:
        new_others[new_other_dict[o]] = others[o]
    return new_others


def filter_ap_edges(new_author_dict, new_paper_dict):
    ap = np.load(old_path + 'author_paper.npy')
    new_ap = []
    for a, p in ap:
        if a in new_author_dict and p in new_paper_dict:
            new_ap.append([new_author_dict[a], new_paper_dict[p]])
    return new_ap


def filter_ff_edges(old_fields, new_field_dict):
    field_parent = np.load(old_path + 'field_parent.npy')
    new_fp = []
    for f, p in field_parent:
        if old_fields[f] in new_field_dict and old_fields[p] in new_field_dict:
            new_fp.append([new_field_dict[old_fields[f]], new_field_dict[old_fields[p]]])
    return new_fp


if __name__ == '__main__':
    old_fields = np.load(old_path + 'fields.npy')
    fields = set(old_fields)
    new_fields = get_selected_fields()
    fields = fields.intersection(new_fields)
    lst = list(fields)
    print(len(lst), 'fields')
    # np.save('../data/fields.npy', lst)
    field_dict = list_to_dict(lst)  # field_id: new_index
    field_author_dict, new_author_dict = filter_other(old_fields, fields, 'author')
    with open(old_path + 'filtered_authors.pickle', 'rb') as f:
        authors = pickle.load(f)
    authors = create_others(new_author_dict, authors)
    # with open('../data/filtered_authors.pickle', 'wb') as f:
    #     pickle.dump(authors, f)
    print(len(authors), 'authors')
    field_paper_dict, new_paper_dict = filter_other(old_fields, fields, 'paper')
    print(len(field_paper_dict), 'abaaba')
    papers = create_others(new_paper_dict, np.load(old_path + 'papers.npy'))
    print(len(new_paper_dict), 'papers')
    # np.save('../data/papers', papers)
    new_ap = filter_ap_edges(new_author_dict, new_paper_dict)
    print(len(new_ap), 'a-p edges')
    # np.save('../data/author_paper', new_ap)
    new_ff = filter_ff_edges(old_fields, field_dict)
    print(len(new_ff), 'f-f edges')
    # np.save('../data/field_parent', new_ff)
    new_af = create_other_field_edges(field_author_dict, new_author_dict, field_dict)
    print(len(new_af), 'a-f edges')
    # np.save('../data/author_field', new_af)
    new_pf = create_other_field_edges(field_paper_dict, new_paper_dict, field_dict)
    print(len(new_pf), 'p-f edges')
    # np.save('../data/paper_field', new_pf)"""
