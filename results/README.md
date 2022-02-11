For each model, we train ten times (from model_0 to model_9) and use process.py to average the results. There are four questions we discuss.

# Prediction_info.csv
This is to show the prediction result (RQ1 in paper).

# Q1_info.csv
This is to show the attention effects at different hit types (RQ2 in paper).

# Q2_info.csv
This is to show the attetion results of model in dataset with different node numbers (three types, >=5<10, >=10<20, >20). However, we find the metric "Hit@k" is related to node number, so it cannot to show the
attention effects at this question. Moerover, for the metric "Hit@%k", we find there are litte differences among samples with different data size. Thus we do not analyse it in paper.

# Q3_info.csv
This is to show how R-GCN-att performs on different CWEs, so that we can analyse what factors can affect the attention results (RQ3 in paper). 
