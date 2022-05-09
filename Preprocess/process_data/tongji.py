import json as js
import csv
with open('data/mapped_final_data.json', 'r')as f:
    a = js.load(f)
    f.close()
s_p, s_n = 0, 0
tongji = []
for cwe in a:
    p = 0
    n = 0
    for d in a[cwe]:
        if d['target'] == 0:
            n += 1
        elif d['target'] == 1:
            p += 1
        else:
            print("wrong")
            exit(-2)
    s_p += p
    s_n += n
    if n == 0:
        print(cwe, p, n)
        tongji.append([cwe, p+n, p, n, ''])
        continue
    print('cwe: ', cwe, p, n, ' ', p/n)
    tongji.append([cwe, p+n, p, n, p/n])
print(s_p, s_n)
print(111)
with open('tongji.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    for dd in tongji:
        writer.writerow(dd)
    writer.writerow('')
    writer.writerow([s_p+s_n, s_p, s_n, s_p/s_n])
    f.close()
