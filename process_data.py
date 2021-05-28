import pymysql
import pickle
import tqdm
import numpy as np


def load_authors(path='data/author_info.pickle'):
    with open(path, 'rb') as f:
        authors = pickle.load(f)
        print(f'{len(authors)} authors')
    return authors


def load_fields(authors, limit=None):
    # connect Acemap
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_analysis'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()

    # query fields
    if limit is not None:
        query = f'select field_id, author_rank_statistic from am_analysis.am_field_analysis limit {limit}'
    else:
        query = 'select field_id, author_rank_statistic from am_analysis.am_field_analysis'
    cursor.execute(query)
    results = cursor.fetchall()
    author_field_id_dict = dict()
    for author in authors:
        author_field_id_dict[author['id']] = []
    for row in results:
        for a in eval(row[1]):
            if a['author_id'] in author_field_id_dict:
                author_field_id_dict[a['author_id']].append(row[0])
                # print(row[0], a['author_id'])
    return author_field_id_dict


def author_paper_edges(authors):
    """Create papers and author-paper edges.

    :return paper_dict: dict(paper_id: index)
    :return papers: list(paper_id)
    :return author_paper: list(list(author, paper))
    """
    paper_dict = dict()  # paper_id: index
    author_paper = []
    count = 0

    for i, author in enumerate(authors):
        for paper_id in author['paper_id']:
            if paper_id not in paper_dict:
                paper_dict[paper_id] = count
                count += 1
            author_paper.append([i, paper_dict[paper_id]])

    print(f'{len(paper_dict)} papers')
    print(f'{len(author_paper)} author-paper edges')

    papers = [-1 for _ in range(count)]
    for paper_id in paper_dict:
        papers[paper_dict[paper_id]] = paper_id

    return paper_dict, papers, author_paper


def author_field_edges(authors, author_field_id_dict):
    field_dict = dict()
    author_field = []
    count = 0

    for i, author in enumerate(authors):
        for field_id in author_field_id_dict[author['id']]:
            if field_id not in field_dict:
                field_dict[field_id] = count
                count += 1
            author_field.append((i, field_dict[field_id]))

    print(f'{len(field_dict)} fields')
    print(f'{len(author_field)} author-field edges')

    fields = [-1 for _ in range(count)]
    for field_id in field_dict:
        fields[field_dict[field_id]] = field_id

    return field_dict, fields, author_field


def paper_field_edges(paper_dict, fields):
    # connect Acemap
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_paper'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()

    paper_field = []
    for i, field_id in enumerate(tqdm.tqdm(fields)):
        query = f"select paper_id from am_paper.am_paper_field where field_id={field_id}"
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            paper_id = row[0]
            if paper_id in paper_dict:
                paper_field.append([paper_dict[paper_id], i])

    print(f'{len(paper_field)} paper-field edges')

    return paper_field


def parent_field(fields):
    # connect Acemap
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_paper'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()

    d = dict()
    for i, field_id in enumerate(fields):
        d[field_id] = i
    parents = []
    for field_id in d:
        query = f'select parent_id from am_paper.am_field_relation where field_id={field_id}'
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            if row[0] in d:
                parents.append([d[field_id], d[row[0]]])

    print(f'{len(parents)} field-field edges')

    return parents


def load_data(path='data'):
    data = dict()
    data['authors'] = load_authors(path + '/author_info.pickle')
    data['papers'] = np.load(path + '/papers.npy')
    data['fields'] = np.load(path + '/fields.npy')
    data['author_paper'] = np.load(path + '/author_paper.npy')
    data['author_field'] = np.load(path + '/author_field.npy')
    data['paper_field'] = np.load(path + '/paper_field.npy')
    return data


if __name__ == '__main__':
    """fields = np.load('data/fields.npy')
    field_field = np.load('data/field_field.npy')
    parent_dict = dict()
    field_parent_dict = defaultdict(list)  # id-id
    count = 0
    for edge in field_field:
        field_id = fields[edge[1]]
        if field_id not in parent_dict:
            parent_dict[field_id] = count
            count += 1
        field_parent_dict[fields[edge[0]]].append(field_id)
    paper_field = np.load('data/paper_field.npy')
    paper_parent = []
    for edge in paper_field:
        field_id = fields[edge[1]]
        for parent_id in field_parent_dict[field_id]:
            paper_parent.append((edge[0], parent_dict[parent_id]))
    print(len(paper_parent))
    paper_parent = list(set(paper_parent))
    print(len(paper_parent))
    parents = [-1 for _ in range(count)]
    for parent in parent_dict:
        parents[parent_dict[parent]] = parent
    np.save('data/only_parent/fields', parents)
    np.save('data/only_parent/paper_field', paper_parent)"""
