"""生成csv文件，以手工标记数据"""
import csv
import json as js

with open('data/processed_test_data.json', 'r')as f:
    data = js.load(f)
    f.close()

test_data = {}

for d in data:
    if d['cwe'] not in test_data:
        test_data[d['cwe']] = [d]
    else:
        test_data[d['cwe']].append(d)

with open('data/flaw_line_info.json', 'r')as f:
    old_label_lines = js.load(f)
    f.close()

# with open('data/mapped_final_data.json', 'r')as f:
#     data = js.load(f)
#     f.close()
#
# with open('data/flaw_line_info.json', 'r')as f:
#     old_label_lines = js.load(f)
#     f.close()
#
# with open('data/before_cross_fun_all_data.json', 'r')as f:
#     before_data = js.load(f)
#     f.close()
examples_number = 0
for cwe in test_data:
    with open('prepare_data/{}.csv'.format(cwe), 'w', newline='')as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'cwe', 'main_file', 'java_file', 'func_name', 'target', 'old_labeled_lines', 'case_id',
                         'bad_source_lines', 'bad_sink_lines', 'fixed_lines'])
        old_lines = old_label_lines[cwe]
        for d in test_data[cwe]:
            func_info = d['fun_info']
            split_info = func_info.split('____')
            java_file_name = split_info[0]
            func_name = split_info[1]
            o_lines = []
            if java_file_name + '.java' in old_lines:
                o_lines = old_lines[java_file_name + '.java']

            all_java_file = []
            for n in d['nodes']:
                if n['java_file'] not in all_java_file:
                    all_java_file.append(n['java_file'])

            for j_file in all_java_file:
                if j_file == java_file_name:
                    writer.writerow(
                        [examples_number, cwe, java_file_name, j_file, func_name, d['target'], o_lines, examples_number])
                else:
                    writer.writerow(
                        [examples_number, cwe, '', j_file, '', '', '', examples_number])
            # if java_file_name.endswith('a'):
            #     new_java_file_name = java_file_name[:-1]
            # else:
            #     new_java_file_name = None
            #
            # if new_java_file_name is not None:
            #     for key in before_cwe_data:
            #         file_name = key.replace('.java', '')
            #         if java_file_name != file_name and new_java_file_name in key:
            #             writer.writerow([examples_number, cwe, file_name, '', '', '', examples_number])
            # else:
            #     examples_number += 1
            #     writer.writerow('')
            #     continue
            #
            examples_number += 1
            writer.writerow('')
    f.close()
