The `labeled_test_data` folder contains many .csv files. They show that how we label the vulnerability-related code lines of samples in test set.
The `process_data` folder contains source code for processing data, detailing the process of converting Juliet to Juliet+.

Next, we will detail the basic dataset Juliet, how we label the vulnerability-related code lines and how to generate the data flow graph DDG.

# 1. Juliet test suite Java1.3 (Juliet)
  This is the basic dataset we used, you can download it through https://samate.nist.gov/SRD/testsuite.php. It should be noted that in a 'mixed' type test case of         Juliet, more than one training example of Juliet+ can be generated.

# 2. Juliet+
  Based on Juliet, we create our dataser named Juliet+, where the vulnerability-related code lines are labeled manunally. The file statistic.csv describe the class         distribution of 70 CWEs in Juliet+. There are 29990 samples, of which 16051 are bad and 13939 are good.
## 2.1 why manually mark vulnerability-related lines of code
   First, some of the vulnerability-related code lines given by Juliet are mislabeled. Second, Juliet does not give the vulnerble code lines which are not at the primary    files (see below for the meaning of "primary"). Third, Juliet does not give the repaired code lines for the good executions. Fourth, we want explore how the models'      interpretability preforms on the different type of code lines. see examples in paper.
## 2.2 how to label the vulnerability-related code lines
   We only label the samples in test set (about 7500). See the CWE190.csv as an example.
   
   "case_id" is the column to show the order of the samples in the test set. The column "cwe" denotes the CWE ID. Each sample has a main (primary) java file (column      "main_file") and at least one java file (column "java_file"). A core function (column “func_name”) that starts the good or bad execution must be put in the main        java file. The column "target" claims that a sample is vulnerable (1) or not (0).
   
   The column "old_label_lines" defines the vulnerability-related code lines given by Juliet. We observe that it does not provide the fixed code lines and contains        some mislabeled lines. To that end, we manually relabel the vulnerability-related code lines in our dataset Juliet+ which generates three columns, i.e.,             "bad_source_lines" and "bad_sink_lines" for each bad sample, and "fixed_lines" for each good sample. 
   
   See the paper for more details (e.g., the meanings of "source" and "sink").

# 3. Generate DDG
  We show the process with the following example.
  ![image](https://github.com/Ng13oTy/Interpretability/blob/main/Preprocess/pictures/example.PNG)
  we first need to generate its CFG
  ![image](https://github.com/Ng13oTy/Interpretability/blob/main/Preprocess/pictures/ctr.PNG)
## preprocess Juliet
We use Javaparser (https://javaparser.org/) to parser the source code of Juliet. To generate the CFG, we refer to the project AISEC (https://github.com/Fraunhofer-AISEC/cpg). The specific steps are contained in the codes of file "process_data".



 
