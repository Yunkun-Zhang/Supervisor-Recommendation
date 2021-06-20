import pickle
import numpy as np
from recommend import print_author_info, get_recommendation, paper_info, authors


title_paper_dict = {}
for p in paper_info:
    title_paper_dict[paper_info[p]['title'].lower()] = p
name_author_dict = {}
author_ids = np.load('data/60+2015+0.5/authors.npy')
for i in author_ids:
    name_author_dict[authors[i]['chinese_name']] = i

if __name__ == '__main__':

    with open('stars.pickle', 'wb') as f:
        pickle.dump([[], []], f)
    with open('stars.pickle', 'rb') as f:
        star_papers, star_authors = pickle.load(f)
    print('=======================================================================')
    print('Welcome to Supervisor Recommendation System!')
    print('Enter:')
    print('  "star" + paper title:          star papers')
    print('  "star" + author name:          star authors')
    print('  "unstar paper" + index/title:  unstar papers')
    print('  "unstar author" + index/name:  unstar authors')
    print('  "show":                        see all the starred papers and authors')
    print('  "aff" + affiliation:           constrain your target school')
    print('  "rec":                         get your supervisor recommendation!')
    print('  "exit":                        exit the system')
    print('=======================================================================')

    aff = ''
    while True:
        x = input('> ')
        if x[:5] == 'star ':
            if x[-4:] == '.txt':
                with open('stars.pickle', 'wb') as f:
                    fin = open(x[5:], 'r')
                    line = fin.readline()[:-1]
                    while line:
                        if line.lower() not in title_paper_dict:
                            print(f'No paper with title "{line}".')
                        else:
                            star_papers.append(line)
                        line = fin.readline()[:-1]
                    pickle.dump([star_papers, star_authors], f)
                    fin.close()
            else:
                if x[5:].lower() not in title_paper_dict:
                    if x[5:] not in name_author_dict:
                        print(f'No paper with title {x[5:]} or author with name {x[5:]}')
                    else:
                        star_authors.append(x[5:])
                    continue
                with open('stars.pickle', 'wb') as f:
                    star_papers.append(x[5:])
                    pickle.dump([star_papers, star_authors], f)
        elif x[:4] == 'show':
            print('Your star papers:')
            for i, title in enumerate(star_papers):
                print(f'  {i+1:<3} {title}')
            print('Your star authors:')
            for i, name in enumerate(star_authors):
                print(f'  {i+1:<3} {name}（{authors[name_author_dict[name]]["affiliation"]}）')
        elif x[:7] == 'unstar ':
            if x[7:] == 'all':
                star_papers = []
                star_authors = []
            elif x[7:13] == 'paper ':
                if len(star_papers) == 0:
                    print('You should star some papers first!')
                try:
                    index = int(x[13:])
                    star_papers.pop(index - 1)
                except:
                    star_papers.remove(x[13:])
            elif x[7:14] == 'author ':
                if len(star_authors) == 0:
                    print('You should star some authors first!')
                try:
                    index = int(x[14:])
                    star_authors.pop(index - 1)
                except:
                    star_authors.remove(x[14:])
        elif x[:4] == 'aff ':
            aff = x[4:]
        elif x == 'rec':
            if len(star_papers) == 0 and len(star_authors) == 0:
                print('You should star some papers or authors first!')
                continue
            if aff == '':
                print('You should specify your target school first!')
                continue
            aid = get_recommendation(aff)
            if not aid:
                print('No such school in our database.')
            print('Your top 3 recommended supervisors:')
            for i, a in enumerate(aid):
                print(f'  {i + 1}', end=' ')
                print_author_info(a)
        elif x == 'exit':
            break
