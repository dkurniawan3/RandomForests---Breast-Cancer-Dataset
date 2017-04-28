# RandomForests w/ Breast Cancer Dataset

Implementation of a random forest classifier with an accuracy of ~95% and OOB of 5%. We are using the UCI Breast Cancer Dataset to  perform binary classification on the dataset to determine if a tumor is benign or malignant. Each line of the dataset describes an instance using 10 columns: the first 9 describe the tumor’s characteristics, and the last column is the ground truth label for the tumor classification. (0 for benign, 1 for malignant).
 
  1. util.py: A file containing utility functions to help build the decision tree.
  2. decision_tree.py: A file containing a decision tree class with learn and classify methods to build the random forest.
  3. random_forest.py: A file containing a random forest class and a main to test the random forest.
