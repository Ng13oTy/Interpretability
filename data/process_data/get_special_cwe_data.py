import json as js

with open('data/labeled_processed_test_data.json', 'r')as f:
    data = js.load(f)
    f.close()
cwes = ['CWE23', 'CWE36', 'CWE80', 'CWE81', 'CWE83', 'CWE89', 'CWE190', 'CWE191']
get_data = {}
for cwe in cwes:
    if cwe not in get_data:
        get_data[cwe] = []

for d in data:
    if d['cwe'] in cwes:
        get_data[d['cwe']].append(d)

dd = []
dd.extend(get_data['CWE23'])
dd.extend(get_data['CWE36'])
with open('data/CWE22.json', 'w')as f:
    js.dump(dd, f)
    f.close()

dd = []
dd.extend(get_data['CWE80'])
dd.extend(get_data['CWE81'])
dd.extend(get_data['CWE83'])

with open('data/CWE79.json', 'w')as f:
    js.dump(dd, f)
    f.close()

for k in get_data.keys():
    if k == 'CWE89' or k == 'CWE190' or k == 'CWE191':
        with open('data/{}.json'.format(k), 'w')as f:
            js.dump(get_data[k], f)
            f.close()
