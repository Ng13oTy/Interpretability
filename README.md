# Interpretability
This is for paper "Can Deep Learning Models Learn the Vulnerable Patterns for Vulnerability Detection?"
For now, we only show part of the label file to show how to label the vulnerability-related code lines for each sample in our dataset. We will publish the dataset,models' implementation and experimental results after our paper is published.

In the file CWE190.csv, "case_id" is the order of the samples in test set which contains about 7500 samples. "cwe" is the cwe ID. For each sample, there is a main (primary) java file ("main_file"), while "java_file" represents which java files this sample involves. In the main java file, there is a main function (func_name) which start the good or bad excution. "target" is to show this sample is vulnerable (1) or not (0).

"old_label_lines" is the vulnerability-related code lines given by Juliet. However, we can see that it only give the lines in primary file. Moreover, it does not give the fixed code lines and there are some lines are mislabeled. So, to meet our needs, we label "bad_source_lines and "bad_sink_lines" for each bad sample while "fixed_lines" for each good sample.
