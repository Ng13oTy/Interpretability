import json as js
from xml.dom import minidom

dom = minidom.parse('data/manifest-109-malFeC.xml')
elementobj = dom.documentElement
subElementObj = elementobj.getElementsByTagName("testcase")

cwe = {}
needed_cwe = [23, 36, 78, 15, 90, 113, 114, 129, 80, 134, 190, 191, 81, 193, 197, 226, 252, 253, 256, 259, 315, 319,
              321, 325, 327, 328, 329, 336, 338, 369, 378, 379, 400, 83, 470, 478, 481, 482, 483, 511, 523, 526, 533,
              534, 535, 539, 549, 563, 566, 572, 597, 598, 600, 601, 605, 606, 613, 614, 615, 643, 674, 681, 690, 759,
              760, 772, 775, 789, 835, 89]
print('total_cwe_number: {}'.format(len(needed_cwe)))
total_testcase_number = 0
for testcase in subElementObj:
    files = testcase.getElementsByTagName("file")
    for file in files:
        maybe_line = file.getElementsByTagName("mixed")
        if len(maybe_line) != 0:
            path = file.getAttribute("path")
            filename = path.split('/')[-1]
            cwe_name = filename.split('_')[0]
            if int(cwe_name.replace('CWE', '')) not in needed_cwe:
                continue
            all_flaw_line = []
            for line in maybe_line:
                all_flaw_line.append(int(line.getAttribute("line")))

            if cwe_name not in cwe:
                cwe[cwe_name] = {filename: all_flaw_line}
            else:
                cwe[cwe_name][filename] = all_flaw_line

for cwe_id in cwe.keys():
    total_testcase_number += len(cwe[cwe_id].keys())

print('total_case: {}'.format(total_testcase_number))
with open('data/flaw_line_info.json', 'w') as f:
    js.dump(cwe, f)
    f.close()
print(111)
