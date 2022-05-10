# Interpretability
This study presents an experimental evaluation to investigate how the different deep learning models predict software vulnerability and if they have recognized the critical code segments representing the software vulnerability. With the help of our proposed framework, researchers understand the predicted results why the source code is considered as a vulnerable one more easily. We list the labeled dataset (about CWE), the source code of our model and the related experiment results in this supplement. `you can reproduce our experiments easily`.

There are two main steps, one is data preprocessing, and the other is model training and testing.

# Preprocess:
It details how we turned Juliet into the data the models received

# Experiments:
You can reproduce our result or train a new model after you read "readme.md" in folder `Experiments`

# Environment
python==3.9

torch==1.11.0

gensim==4.1.2

scikit-learn==1.0.

dgl==11.1(CUDA)

