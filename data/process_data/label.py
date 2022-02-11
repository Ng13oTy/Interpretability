import csv
import json as js
import os

with open('data/processed_test_data.json', 'r')as f:
    before_data = js.load(f)
    f.close()

label_data = {}

for f in os.listdir('prepare_data_copy'):
    with open('prepare_data_copy/' + f)as ff:
        data = list(csv.reader(ff))[1:]
        for line in data:
            if len(line) < 11:
                continue
            cwe = line[1]
            if cwe == '':
                continue
            if cwe not in label_data:
                label_data[cwe] = {}
            case_id = line[0]
            if case_id not in label_data[cwe]:
                label_data[cwe][case_id] = {}
            main_file = line[2]
            if main_file != '':
                label_data[cwe][case_id]['main_file'] = main_file
                label_data[cwe][case_id]['func_name'] = line[4]
            bad_source_line = line[8]
            bad_sink_line = line[9]
            fixed_line = line[10]
            if bad_source_line != '':
                label_data[cwe][case_id]['bad_source'] = [line[3], bad_source_line]
            if bad_sink_line != '':
                label_data[cwe][case_id]['bad_sink'] = [line[3], bad_sink_line]
            if fixed_line != '':
                label_data[cwe][case_id]['fixed'] = [line[3], fixed_line]
        ff.close()

for b_d in before_data:
    split_info = b_d['fun_info'].split('____')
    m_file = split_info[0]
    m_func = split_info[1]
    label_info = label_data[b_d['cwe']]
    for k in label_info:
        if label_info[k]['main_file'] == m_file and label_info[k]['func_name'] == m_func:
            if 'bad_source' in label_info[k]:
                for n in b_d['nodes']:
                    if n['java_file'] == label_info[k]['bad_source'][0] and n['line'] == int(label_info[k]['bad_source'][1]):
                        n['bad_source'] = True

            if 'bad_sink' in label_info[k]:
                for n in b_d['nodes']:
                    if n['java_file'] == label_info[k]['bad_sink'][0] and n['line'] == int(label_info[k]['bad_sink'][1]):
                        n['bad_sink'] = True

            if 'fixed' in label_info[k]:
                for n in b_d['nodes']:
                    if n['java_file'] == label_info[k]['fixed'][0] and n['line'] == int(label_info[k]['fixed'][1]):
                        n['fixed'] = True

            break
with open('data/labeled_processed_test_data.json', 'w')as f:
    js.dump(before_data, f)
    f.close()
print(111)


