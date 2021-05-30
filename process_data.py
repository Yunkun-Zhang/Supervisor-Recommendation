import pymysql
import pickle
import tqdm
import numpy as np
from collections import defaultdict


def load_authors(path='data/author_info.pickle'):
    with open(path, 'rb') as f:
        authors = pickle.load(f)
    return authors


def list_to_dict(lst):
    d = dict()
    for idx, item in enumerate(lst):
        d[item] = idx
    return d


"""def top_authors(authors, num_aff=50, num_author=20):
    aff_dict = defaultdict(list)
    for author in authors:
        aff_dict[author['affiliation']].append(author)
    sort = sorted(aff_dict.keys(), key=lambda a: len(aff_dict[a]), reverse=True)
    authors = []
    for a in sort[:num_aff]:
        a_sort = sorted(aff_dict[a], key=lambda at: len(at['paper_id']), reverse=True)
        authors.extend(a_sort[:min(num_author, len(a_sort))])
    print(f'{len(authors)} authors')
    return authors"""


def filter_paper(paper_dict):
    # connect Acemap
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_analysis'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()
    print('Query papers...')
    query = 'SELECT am_analysis.am_paper_analysis.paper_id FROM am_analysis.am_paper_analysis INNER JOIN ' \
            'am_paper.am_paper on am_analysis.am_paper_analysis.paper_id = am_paper.am_paper.paper_id ' \
            'where reference_count > 60 and year > 2000'
    cursor.execute(query)
    results = cursor.fetchall()
    new_paper_dict = dict()
    papers = []
    count = 0
    for row in results:
        if row[0] in paper_dict:
            new_paper_dict[row[0]] = count
            count += 1
            papers.append(row[0])
    print(f'{len(new_paper_dict)} papers')
    return new_paper_dict, papers  # paper_id: count


def filter_author_by_paper(authors, papers, author_paper, new_paper_dict):
    author_dict = dict()
    new_author_paper = []
    count = 0
    for a, p in tqdm.tqdm(author_paper, desc='Filter authors'):
        if papers[p] in new_paper_dict:
            if a not in author_dict:
                author_dict[a] = count
                count += 1
            new_author_paper.append([author_dict[a], new_paper_dict[papers[p]]])
    new_authors = [None for _ in range(count)]
    for a in author_dict:
        new_authors[author_dict[a]] = authors[a]
    print(f'{len(new_authors)} authors')
    print(f'{len(new_author_paper)} author-paper edges')
    return new_authors, new_author_paper


def load_fields(authors, max_level=3, limit=None):
    """Search for all fields with level <= 3 linking the authors."""

    # connect Acemap
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_analysis'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()

    # query fields
    query = 'select am_paper.am_field.field_id, am_paper.am_field.level,' \
            ' am_analysis.am_field_analysis.author_rank_statistic from am_analysis.am_field_analysis' \
            ' inner join am_paper.am_field on am_analysis.am_field_analysis.field_id =' \
            ' am_paper.am_field.field_id'
    if limit is not None:
        query += f' limit {limit}'
    print('Query fields...')
    cursor.execute(query)
    results = cursor.fetchall()

    author_field_level_dict = dict()
    for author in authors:
        author_field_level_dict[author['id']] = []

    for field_id, level, author in tqdm.tqdm(results, desc='Adding fields'):
        for a in eval(author):
            if a['author_id'] in author_field_level_dict:
                if level <= max_level:
                    author_field_level_dict[a['author_id']].append((field_id, level))
                fields = [field_id]
                while level > max_level and fields:
                    query = 'select am_paper.am_field_relation.parent_id, am_paper.am_field.level from' \
                            ' am_paper.am_field_relation inner join am_paper.am_field on' \
                            ' am_paper.am_field_relation.parent_id = am_paper.am_field.field_id' \
                            ' where am_paper.am_field.field_id'
                    if len(fields) == 1:
                        query += f'={fields[0]}'
                    else:
                        query += f' in{tuple(fields)}'
                    cursor.execute(query)
                    res = cursor.fetchall()
                    fields = []
                    for pf, pl in res:
                        if pl <= max_level:
                            author_field_level_dict[a['author_id']].append((pf, pl))
                        else:
                            fields.append(pf)
                    level -= 1

    return author_field_level_dict


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


def author_field_edges(authors, author_field_level_dict):
    """Add edges from existing fields."""

    field_dict = dict()
    author_field = []
    count = 0

    for i, author in enumerate(authors):
        for field_id, level in author_field_level_dict[author['id']]:
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
    for i, field_id in enumerate(tqdm.tqdm(fields, desc='Adding paper-field edges')):
        query = f"select paper_id from am_paper.am_paper_field where field_id={field_id}"
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            paper_id = row[0]
            if paper_id in paper_dict:
                paper_field.append([paper_dict[paper_id], i])

    print(f'{len(paper_field)} paper-field edges')

    return paper_field


def parent_field(field_dict, fields):
    """Look for parent fields (not necessarily in original fields)."""
    # connect Acemap
    acemap = {'user': 'mobilenet',
              'passwd': 'mobilenet',
              'host': '202.120.36.29',
              'port': 13307,
              'database': 'am_paper'}

    db = pymysql.connect(**acemap)
    cursor = db.cursor()

    count = len(fields)
    field_parents = []
    query_fields = fields[:]
    while query_fields:
        for field_id in tqdm.tqdm(query_fields, desc='Query parent fields'):
            query = f'select am_paper.am_field.level, am_paper.am_field_relation.parent_id from' \
                    f' am_paper.am_field inner join am_paper.am_field_relation on am_paper.am_field.field_id' \
                    f' = am_paper.am_field_relation.parent_id where am_paper.am_field_relation.field_id={field_id}'
            cursor.execute(query)
            results = cursor.fetchall()
            query_fields = []
            for level, parent in results:
                if parent not in field_dict:
                    field_dict[parent] = count
                    count += 1
                    fields.append(parent)
                field_parents.append([field_dict[field_id], field_dict[parent]])
                if level > 1:
                    query_fields.append(parent)

    print(f'{len(fields)} fields in total')
    print(f'{len(field_parents)} field-field edges')

    return field_dict, fields, field_parents


def load_data(path='data'):
    data = dict()
    data['authors'] = load_authors(path + '/filtered_authors.pickle')
    data['papers'] = np.load(path + '/papers.npy')
    data['fields'] = np.load(path + '/fields.npy')
    data['author_paper'] = np.load(path + '/author_paper.npy')
    data['author_field'] = np.load(path + '/author_field.npy')
    data['paper_field'] = np.load(path + '/paper_field.npy')
    data['field_parent'] = np.load(path + '/field_parent.npy')
    print(f'{len(data["authors"])} authors,',
          f'{len(data["papers"])} papers,',
          f'{len(data["fields"])} fields')
    print(f'{len(data["author_paper"])} author-paper edges,',
          f'{len(data["author_field"])} author-field edges,',
          f'{len(data["paper_field"])} paper-field edges,',
          f'{len(data["field_parent"])} field-parent edges')
    return data


def process(data):
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
    for f1, f2 in data['field_parent']:
        field_field[f1].append(f2)
        field_field[f2].append(f1)

    # create metapaths
    p_a_p = []
    for a, p_list in tqdm.tqdm(author_paper.items(), desc='Meta-path p-a-p'):
        p_a_p.extend([(p1, a, p2) for p1 in p_list for p2 in p_list])
    p_a_p = np.array(p_a_p)
    p_a_p[:, 1] += num_paper

    """f_f_f = []
    for f, f_list in tqdm.tqdm(field_field.items(), desc='Meta-path f-f-f'):
        f_f_f.extend([(f1, f, f2) for f1 in f_list for f2 in f_list])
    f_f_f = np.array(f_f_f)
    f_f_f += num_paper + num_author"""

    a_f_a = []
    for f, a_list in tqdm.tqdm(field_author.items(), desc='Meta-path a-f-a'):
        a_f_a.extend([(a1, f, a2) for a1 in a_list for a2 in a_list])
    a_f_a = np.array(a_f_a)
    a_f_a[:, 1] += num_paper

    # save metapaths
    expected_metapaths = [
        [(0, 1, 0)],
        [(1, 2, 1)]
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


if __name__ == '__main__':
    # authors and papers
    """authors = load_authors()
    paper_dict, papers, author_paper = author_paper_edges(authors)
    new_paper_dict, new_papers = filter_paper(paper_dict)
    np.save('data/papers', new_papers)
    authors, author_paper = filter_author_by_paper(authors, papers, author_paper, new_paper_dict)
    with open('data/filtered_authors.pickle', 'wb') as f:
        pickle.dump(authors, f)
    np.save('data/author_paper', author_paper)"""

    # fields
    """authors = load_authors('data/filtered_authors.pickle')
    author_field_level_dict = load_fields(authors)
    field_dict, fields, author_field = author_field_edges(authors, author_field_level_dict)
    np.save('data/author_field', author_field)
    field_dict, fields, field_parent = parent_field(field_dict, fields)
    np.save('data/fields', fields)
    np.save('data/field_parent', field_parent)
    fields = np.load('data/fields.npy')
    papers = np.load('data/papers.npy')
    paper_dict = list_to_dict(papers)
    paper_field = paper_field_edges(paper_dict, fields)
    np.save('data/paper_field', paper_field)"""
