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
  
  We first need to generate its CFG
  
  ![image](https://github.com/Ng13oTy/Interpretability/blob/main/Preprocess/pictures/ctr.PNG)
  
  Note that there are three varibles 'a', 'b', 'r'. To generate DDG, we need to know what variables are read and what variables are written with each node.
  
  ![image](https://github.com/Ng13oTy/Interpretability/blob/main/Preprocess/pictures/wr.PNG)
  
  Next, we draw each data dependency. A data dependency goes from a node that writes into a variable to another node that reads from the variable.To have a valid dependency, we must identify the correct ‘write’ node for each ‘read’ node. That is done as follows.
  
  * Start with a node that reads from a variable. For example, node 3 in the example reads variable a. That read operation is the endpoint of a data dependency.
  * Next, walk backward in the control flow graph until you find a node that writes the same variable. That is the starting point of a data dependency. For example, going backward from node 3, we visit node 2, and then node 1. Only node 1 writes a. Therefore, the data dependency for a goes from 1 to 3.
  * The search for data dependencies has to be performed breadth-first and consider every control path that leads to the endpoint of the data dependency. For example, the read of r in node 5 could be done by node 4 or else by node 3. This is because there is a control flow from node 3 to node 5 5 as well as from node 4 to node 5.
  
  ![image](https://github.com/Ng13oTy/Interpretability/blob/main/Preprocess/pictures/dfg.PNG)
  
  see this website (https://schaumont.dyn.wpi.edu/ece4530f19/lectures/lecture18-notes.html#other-constructions) for more introduction.
  
  We use Javaparser (https://javaparser.org/) to parser the source code of Juliet. To generate the CFG, we refer to the project AISEC (https://github.com/Fraunhofer-AISEC/cpg). The specific steps are contained in the codes of file "process_data".



 
