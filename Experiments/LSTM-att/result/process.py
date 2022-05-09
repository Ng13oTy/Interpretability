import csv
import numpy as np

pre_info = []
q1_info = []
q2_info = []
q3_info = []

for i in range(10):
    with open('model_{}/prediction_info.csv'.format(i), 'r')as f:
        pre = list(csv.reader(f))
        pre_info.append(pre)
        f.close()
    with open('model_{}/Q1_info.csv'.format(i), 'r')as f:
        q1_info.append(list(csv.reader(f)))
        f.close()
    with open('model_{}/Q2_info.csv'.format(i), 'r')as f:
        q2_info.append(list(csv.reader(f)))
        f.close()
    with open('model_{}/Q3_info.csv'.format(i), 'r')as f:
        q3_info.append(list(csv.reader(f)))
        f.close()

new_pre_info = []
new_q1_info = []
new_q2_info = []
new_q3_info = []
for i in range(6):
    temp = []
    for j in range(10):
        temp.append(float(pre_info[j][1][i]))
    new_pre_info.append(temp)
for i in range(1, 4):
    temp1 = []
    for j in range(1, 5):
        temp2 = []
        for k in range(10):
            if q1_info[k][i][j] == '-':
                temp2.append(0)
            else:
                temp2.append(float(q1_info[k][i][j]))
        temp1.append(temp2)
    new_q1_info.append(temp1)

for i in range(1, 10):
    temp1 = []
    for j in range(2, 8):
        temp2 = []
        for k in range(10):
            if q2_info[k][i][j] == '-':
                temp2.append(0)
            else:
                temp2.append(float(q2_info[k][i][j]))
        temp1.append(temp2)
    new_q2_info.append(temp1)

for i in range(1, 26):
    temp1 = []
    for j in range(2, 6):
        temp2 = []
        for k in range(10):
            if q3_info[k][i][j] == '-':
                temp2.append(0)
            else:
                temp2.append(float(q3_info[k][i][j]))
        temp1.append(temp2)
    new_q3_info.append(temp1)

new_pre_info = np.array(new_pre_info)
new_q1_info = np.array(new_q1_info)
new_q2_info = np.array(new_q2_info)
new_q3_info = np.array(new_q3_info)


avg_pre = np.average(new_pre_info, axis=1)

std_q1 = np.std(new_q1_info, axis=2)
avg_q1 = np.average(new_q1_info, axis=2)

std_q2 = np.std(new_q2_info, axis=2)
avg_q2 = np.average(new_q2_info, axis=2)

std_q3 = np.std(new_q3_info, axis=2)
avg_q3 = np.average(new_q3_info, axis=2)

with open('pre_info_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['pre', 'acc', 'recall', 'f1', 'FPR', 'FNR'])
    writer.writerow([round(d, 2) for d in avg_pre])
    f.close()
with open('Q1_info_avg_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
    for i in range(len(avg_q1)):
        writer.writerow([round(d*100, 2) for d in avg_q1[i]])
with open('Q1_info_std_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
    for i in range(len(std_q1)):
        writer.writerow([round(d*100, 3) for d in std_q1[i]])

with open('Q2_info_avg_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['Hit10', 'Hit5', 'Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
    for i in range(len(avg_q2)):
        writer.writerow([round(d*100, 2) for d in avg_q2[i]])
with open('Q2_info_std_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['Hit10', 'Hit5', 'Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
    for i in range(len(std_q2)):
        writer.writerow([round(d*100, 3) for d in std_q2[i]])

with open('Q3_info_avg_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
    for i in range(len(avg_q3)):
        writer.writerow([round(d*100, 2) for d in avg_q3[i]])
with open('Q3_info_std_all.csv', 'w', newline='')as f:
    writer = csv.writer(f)
    writer.writerow(['Hit3', 'Hit1', 'Hit50%', 'Hit30%'])
    for i in range(len(std_q3)):
        writer.writerow([round(d*100, 3)for d in std_q3[i]])

print(11)
