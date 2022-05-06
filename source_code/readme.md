# 1. Directory Structure
There are four models, i.e., LSTM-att, SAN, HAN and R-GCN-att, and all of them use the same data which are placed in the folder "shared_data".
## 1.1 The files in the folder "shared_data"
   * CWE190.json: a sub-dataset of vulnerability type CWE190 in the test set
   * CWE191.json: a sub-dataset of vulnerability type CWE191 in the test set
   * CWE22.json: a sub-dataset of vulnerability type CWE22 in the test set
   * CWE79.json: a sub-dataset of vulnerability type CWE79 in the test set
   * CWE89.json: a sub-dataset of vulnerability type CWE89 in the test set
   * labeled_processed_test_data.json: the labled test set
   * node_type_dic.json: the map of all the node types (7 in total) in DDG
   * w2v.model: the trained word2vec model with corpus of all samples.
   
   Note that we do not publish the preprocessed training set for the time being. These files are used to reproduce our experimental results. To do this, 
   copy these files into the folder named "data" of each model.
## 1.2 The structure of model folder
   All folders that hold the four models have a similar structure
   >* data(folder): save data copied from "shared_data" folder
   >* model(folder): model specific implementation
   >* result(folder):
   >>* model_?
   >>>* {model}.bin: the trained model parameters. show_result.py use it to reproduce our experimental results
   >>>* Q1_info.csv
   >>>* Q2_info.csv
   >>>* Q3_info.csv
   >>>* prediction_info.csv
   >>* ...
   >>* process.py
   >* show_result(folder)
   >* show_result.py
   >* train_test.py
