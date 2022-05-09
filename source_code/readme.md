# 1. Experimental Design
Since many existing state-of-the-art vulnerability detection researches such as [VulDeePecker](https://arxiv.org/abs/1801.01681) and [SySeVR](https://arxiv.org/pdf/1807.06756.pdf) do not use attention mechanism, we cannot use these models directly for experiments. We observe that vulnerability detection models are broadly classified into two categories, namely sequence-based and graph neural network-based. We also consider what source code has in common with natural language. Therefore, we borrow several applications of attention mechanism in NLP field and implement two types of `four vulnerability detection models with attention mechanism`. We devise four questions to investigate the interpretability of the model.

For the reliability of the experimental results, we get `ten` trained results for each model

## 1.1 Model
   * [LSTM-att](https://aclanthology.org/P16-2034.pdf):(Google Scholar 1500+ citations). It itself is to classify the relationship of a sentence. We regard a node in        the DDG as a word, and the entire DDG as a sentence to observe the importance of a certain node.
   * [HAN](https://aclanthology.org/N16-1174.pdf):(Google Scholar 4000+ citations). It classifies documents, taking into account the weight of each word in a sentence      and the weight of each sentence in the document. We regard the source code of each node of DDG as a sentence, and the token in the node as a word, so that each        token has different contributions to the semantic information of the node. In addition, we treat the DDG as a document, each node as a sentence, and observe the        weight of each node.
   * [SAN]: SAN is the same as HAN except that it does not use the attention mechanism to obtain the semantic information of nodes.
   * [R-GCN-att](https://arxiv.org/abs/2109.02527): Mainly refer to its calculation method of graph node weight. The topology information and semantic information of        the graph are considered at the same time
## 1.2 Model paremeters
   see the code in the folder "model" or the introduction in paper
## 1.3 RQ
   * Prediction result: Our motivation is that it only makes sense to analyze the interpretability of a model if it achieves good predictive performance.
   * RQ1: Attention result. On the basis of Experiment 1, observe the overall interpretability of the four models.
   * RQ2: Attention effect on DDGs with different sizes of nodes. This is to analyze whether the number of nodes in the sample affects model interpretability. The test      set is divided into three subsets according to the number of nodes, namely >=5<10,>=10<20,>=20. The three subsets have a uniform distribution of sample numbers.
   * RQ3: Attention effect on different CWEs. Due to the differences in the number of samples and features of different CWEs, we would like to further analyze which        factors will affect the attention effect.

# 2. Directory Structure
There are four models, i.e., LSTM-att, SAN, HAN and R-GCN-att, and all of them use the same data which are placed in the folder "shared_data".
## 2.1 The files in the folder "shared_data"
   * CWE190.json: a sub-dataset of vulnerability type CWE190 in the test set
   * CWE191.json: a sub-dataset of vulnerability type CWE191 in the test set
   * CWE22.json: a sub-dataset of vulnerability type CWE22 in the test set
   * CWE79.json: a sub-dataset of vulnerability type CWE79 in the test set
   * CWE89.json: a sub-dataset of vulnerability type CWE89 in the test set
   * labeled_processed_test_data.json: the labled test set
   * node_type_dic.json: the map of all the node types (7 in total) in DDG
   * w2v.model: the trained word2vec model with corpus of all samples.
   
   Note that we do not publish the `preprocessed` training set for the time being. These files are used to reproduce our experimental results. To do this, 
   copy these files into the folder named "data" of each model.
## 2.2 The structure of model folder
   All folders that hold the four models have a similar structure
   >* data(folder): save data copied from "shared_data" folder
   >* model(folder): model specific implementation
   >* result(folder): store ten trained models and their results
   >>* model_?(0-9): the ten trained result
   >>>* {model}.bin: the trained model parameters. show_result.py use it to reproduce our experimental results
   >>>* Q1_info.csv: result of RQ1
   >>>* Q2_info.csv: result of RQ2
   >>>* Q3_info.csv: result of RQ3
   >>>* prediction_info.csv: result of prediction
   >>* ...
   >>* process.py: to summarize the results of ten trained models
   >* show_result(folder): to save the reproduced experimental results
   >* show_result.py: to reproduce our experimental results
   >* train_test.py: to train and test
# 3. Reproduce Experimental Results
  * install the necessary environment
  * copy the files in the folder "shared_data" to the folder "data"
  * choose any trained model, such as SAN.bin in the `SAN/result/model_0/SAN.bin`
  * run show_result.py (set 'trained_model_path' correctly first) and then you will see the result in the folder "show_result"
  * after get ten results, you can calculate their average result (refer to process.py)
  Note that results may vary slightly due to machine differences
