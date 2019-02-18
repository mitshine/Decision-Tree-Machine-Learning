##########################
#
#  Author: Mitesh Khadgi
#    Date: 02/17/2019
#
##########################

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import ast
import copy
import math
import random

#Copy the folder "data_sets1" and "data_sets2" and this file "dtree.py" in the current folder.
#From the current folder, run the below commands to simulate.
#Use command: python dtree.py <L> <K> <training-set> <validation-set> <test-set> <to-print>
#For data_set1, use command: python dtree.py "3" "5" "data_sets1/training_set.csv" "data_sets1/test_set.csv" "data_sets1/validation_set.csv" "yes"
#For data_set2, use command: python dtree.py "3" "5" "data_sets2/training_set.csv" "data_sets2/test_set.csv" "data_sets2/validation_set.csv" "yes"

#To save the stdout (terminal output) results to a file name "STANDARD_OUTPUT_RESULTS.txt". Uncomment the below block and the line "sys.stdout.close()" at the end of this entire program.
#The below code will create "STANDARD_OUTPUT_RESULTS.txt" file as the base file and create a new file with incrementing number as "STANDARD_OUTPUT_RESULTS<incrementing_number>.txt".
"""
fileName = "STANDARD_OUTPUT_RESULTS_.txt"

try:
    file = open(fileName, 'r')
except IOError:
    file = open(fileName, 'w')

if os.path.isfile(fileName):
    incNum = 0
    while True:
        incNum += 1
        newFileName = fileName.split(".txt")[0] + str(incNum) + ".txt"
        if os.path.isfile(newFileName):
            print("\n%s already exists." % newFileName)
            continue
        else:
            fileName = newFileName
            print("\nCreated new filename: %s" % fileName)
            break

sys.stdout=open(fileName,"w")
"""

#Initialize CountOfNodes (for nodeID) as '0' and to be used later as a global variable.
CountOfNodes = 0

#To ignore all warnings on the stdout (terminal output), uncomment below line.
#UserWarning: Boolean Series key will be reindexed to match DataFrame index.
warnings.simplefilter("ignore")

class Node():
    def __init__(self):
        self.LEFT = None
	self.RIGHT = None
	#attribute represents the attribute name.
	self.attribute = None
	#nodeType represents leaf node ('L') or a root node ('R') or an intermediate node ('I').
	self.nodeType = None
	#value represents split value of an attribute as either '0' or '1'.
	self.value = None
	#posCount and negCount represents '+' and '-' for gain computation.
	self.posCount = None
	self.negCount = None
	#label represents class value of an attribute.
	self.label = None
	#nodeID represents Id of a node,
	self.nodeID = None
    
    def setNodeValue(self, attribute, nodeType, value = None, posCount = None, negCount = None):
        self.attribute = attribute
        self.nodeType = nodeType
        self.value = value
        self.posCount = posCount
        self.negCount = negCount

class Tree():
    def __init__(self, df):
	self.root = Node()
	self.root.setNodeValue('', 'R')
	self.df = df
        
    def createDecisionTree(self, data, tree, heuristic):
        global CountOfNodes
        rows = data.shape[0]
        numOfOnes = data['Class'].sum()
        numOfZeros = rows - numOfOnes        
        if data.shape[1] == 1 or rows == numOfOnes or rows == numOfZeros:
            tree.nodeType = 'L'
            if numOfZeros > numOfOnes:
                tree.label = 0
            else:
                tree.label = 1
            return        
        else:        
            bestAttribute = findAttributeWinner(data, heuristic)
            tree.LEFT = Node()
            tree.RIGHT = Node()
            
            tree.LEFT.nodeID = CountOfNodes
            CountOfNodes = CountOfNodes+1
            tree.RIGHT.nodeID = CountOfNodes
            CountOfNodes = CountOfNodes+1
            
            tree.LEFT.setNodeValue(bestAttribute, 'I', 0, data[(data[bestAttribute]==0) & (self.df['Class']==1) ].shape[0], data[(data[bestAttribute]==0) & (self.df['Class']==0) ].shape[0])
            tree.RIGHT.setNodeValue(bestAttribute, 'I', 1, data[(data[bestAttribute]==1) & (self.df['Class']==1) ].shape[0], data[(data[bestAttribute]==1) & (self.df['Class']==0) ].shape[0])
            self.createDecisionTree( data[data[bestAttribute]==0].drop([bestAttribute], axis=1), tree.LEFT, heuristic)
            self.createDecisionTree( data[data[bestAttribute]==1].drop([bestAttribute], axis=1), tree.RIGHT, heuristic)
            
    def printDecisionTreeLevels(self, node,level):
        a = ""
        if(node.LEFT is None and node.RIGHT is not None):
            for i in range(0,level):    
                a += "| "
            level = level + 1
            b = "{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else ""))
            a += b
            print(a)
            self.printDecisionTreeLevels(node.RIGHT,level)
        elif(node.RIGHT is None and node.LEFT is not None):
            for i in range(0,level):    
                a += "| "
            level = level + 1
            b = "{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else ""))
            a += b
            print(a)
            self.printDecisionTreeLevels(node.LEFT,level)
        elif(node.RIGHT is None and node.LEFT is None):
            for i in range(0,level):    
                a += "| "
            level = level + 1
            b = "{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else ""))
            a += b
            print(a)
        else:
            for i in range(0,level):    
                a += "| "
            level = level + 1
            b = "{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else ""))
            a += b
            print(a)
            self.printDecisionTreeLevels(node.LEFT,level)
            self.printDecisionTreeLevels(node.RIGHT,level)
    
    def printDecisionTree(self, node):
        self.printDecisionTreeLevels(node.LEFT,0)
        self.printDecisionTreeLevels(node.RIGHT,0)
    
    def predictDecisionTreeLabel(self, data, root):
        if root.label is not None:
            return root.label
        elif data[root.LEFT.attribute][data.index.tolist()[0]] == 1:
            return self.predictDecisionTreeLabel(data, root.RIGHT)
        else:
            return self.predictDecisionTreeLabel(data, root.LEFT)

    def countNumOfNodes(self,node):
        if(node.LEFT is not None and node.RIGHT is not None):
            return 2 + self.countNumOfNodes(node.LEFT) + self.countNumOfNodes(node.RIGHT)
        return 0

    def countNumOfLeafs(self,node):
        if(node.LEFT is None and node.RIGHT is None):
            return 1
        return self.countNumOfLeafs(node.LEFT) + self.countNumOfLeafs(node.RIGHT)

def calcVarianceImpurity(labels):
    rows = labels.shape[0]
    numOfOnes = labels.sum().sum()
    numOfZeros = rows - numOfOnes
    numOfOnesCount = 1.0
    numOfZerosCount = 1.0
    if(rows != 0):
        numOfOnesCount = float(numOfOnes)/float(rows)
        numOfZerosCount = float(numOfZeros)/float(rows)
    variance_impurity = numOfOnesCount * numOfZerosCount
    return variance_impurity

def calcEntropy(labels):
    rows = labels.shape[0]
    numOfOnes = labels.sum().sum()
    numOfZeros = rows - numOfOnes
    numOfOnesCount = 0.0
    numOfZerosCount = 0.0
    if(rows != 0):
        numOfOnesCount = float(numOfOnes)/float(rows)
        numOfZerosCount = float(numOfZeros)/float(rows)
    if rows == numOfOnes or rows == numOfZeros:
        return 0
    if((numOfOnesCount == 0) or (numOfZerosCount == 0)):
        entropy = 0
    elif(numOfOnes == numOfZeros):
        entropy = 1
    else:
        entropy = -(numOfOnesCount)*math.log(numOfOnesCount, 2) - (numOfZerosCount)*math.log(numOfZerosCount,2)
    return entropy

def calcGain(fLabels, heuristic):
    rows = fLabels.shape[0]
    numOfOnes = fLabels[fLabels[fLabels.columns[0]] == 1].shape[0]
    numOfZeros = fLabels[fLabels[fLabels.columns[0]] == 0].shape[0]
    oldVal = heuristic(fLabels[['Class']])
    Ones = heuristic(fLabels[fLabels[fLabels.columns[0]] == 1][['Class']])
    Zeros = heuristic(fLabels[fLabels[fLabels.columns[0]] == 0][['Class']])
    infoGain = oldVal - (float(numOfOnes)/float(rows))*Ones - (float(numOfZeros)/float(rows))*Zeros
    return infoGain

def findAttributeWinner(data, heuristic):
    maxGain = -1.0
    for x in data.columns:
        if x == 'Class':
            continue
        currentGain = calcGain(data[[x, 'Class']], heuristic)
        if maxGain < currentGain:
            maxGain = currentGain
            bestAttribute = x
    return bestAttribute
		
def removeNode(tree, x):
    tmpVar = None
    res = None
    if(tree.nodeType != "L"):
        if(tree.nodeID == x):
            return tree
        else:
            res = removeNode(tree.LEFT,x)
            if (res is None):
                res = removeNode(tree.RIGHT,x)
            return res
    else:
        return tmpVar

def pruneTreeFunction(newTree, l, k, data):
    tree_best = newTree
    tree_copy = None
    best_accuracy = calculateAccuracy(data, newTree)
    for i in range(1, l):
        tree_copy = copy.deepcopy(newTree)
        best_accuracy = calculateAccuracy(data, tree_copy)
        m = random.randint(1, k)
        for j in range(1,m):
            x = random.randint(1,tree_copy.countNumOfNodes(tree_copy.root)-1)
            tmpNode = Node()
            tmpNode = removeNode(tree_copy.root,x)

            if(tmpNode is not None):
                tmpNode.LEFT = None
                tmpNode.RIGHT = None
                tmpNode.nodeType = "L"
                if(tmpNode.negCount > tmpNode.posCount):
                    tmpNode.label = 0
                else:
                    tmpNode.label = 1
            curr_accuracy = calculateAccuracy(data, tree_copy)
            if (curr_accuracy > best_accuracy):
                best_accuracy = curr_accuracy
                tree_best = copy.deepcopy(tree_copy)
            else:
                tree_copy = copy.deepcopy(tree_best)
                best_accuracy = calculateAccuracy(data, tree_copy)
    return tree_best, best_accuracy

def calculateAccuracy(data, tree):
    correctCount = 0
    for i in data.index:
        val = tree.predictDecisionTreeLabel(data.iloc[i:i+1, :].drop(['Class'], axis=1),tree.root)
        if val == data['Class'][i]:
            correctCount = correctCount + 1
    var = 100*(float(correctCount)/float(data.shape[0]))
    return var

def main():
	args = str(sys.argv)
	#ast.literal_eval is used to check if the input argument is a valid python datatype.
	args = ast.literal_eval(args)
	
	if (len(args) < 6):
		print("Total number of required input arguments should be 6. Check README.txt file for input format.\n")
	elif (args[3][-4:] != ".csv" or args[4][-4:] != ".csv" or args[5][-4:] != ".csv"):
		print(args[2])
		print("Check your training, validation and test file. It must be a .csv !\n")
	else:
		l_initial = int(args[1])
		k_initial = int(args[2])
		training_set = str(args[3])
		validation_set = str(args[4])
		test_set = str(args[5])
		to_print = str(args[6])
	
	df = pd.read_csv(training_set)
	dtest = pd.read_csv(test_set)
	dvalidation = pd.read_csv(validation_set)

	#Remove empty rows
	df = df.dropna()
	dtest = dtest.dropna()
	dvalidation = dvalidation.dropna()

	#Variation 1:
	#l_array = [2, 5, 7, 9, 10, 13, 15, 17, 18, 19]
	#k_array = [9, 10, 20, 13, 15, 19, 20, 19, 20, 21]

	#Variation 2:
	#l_array = [2, 15, 7, 50, 40, 35, 23, 55, 18, 1]
	#k_array = [10, 1, 20, 20, 60, 40, 21, 55, 25, 35]

	#Variation 3: Takes much longer time to complete.
	#l_array = [60, 15, 20, 25, 30, 35, 40, 45, 50, 55]
	#k_array = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

	#Variation 4: Takes lesser simulation time.
	l_array = [2, 2, 7, 5, 4, 3, 3, 2, 1, 1]
	k_array = [1, 2, 2, 2, 6, 4, 2, 3, 2, 3]

	print("\nStarted simulation at 0 seconds\n")
	print("Please wait to complete building the Decision Tree using Information Gain Heuristic!\n")
	start = time.time()

	#Build a decision tree using Information Gain Heuristic.
	dtree = Tree(df)
	dtree.createDecisionTree(df, dtree.root, heuristic=calcEntropy)
	#Calculate accuracy of the decision tree on validation data set.
	maxAccuracy = calculateAccuracy(dvalidation, dtree)
	#Save it as the best decision tree.
	bestTree = copy.deepcopy(dtree)
	#Count the number of nodes in the best decision tree.
	countOfNodes = bestTree.countNumOfNodes(bestTree.root)

	print("-------------------------------------------")
	print("Build Tree using Information Gain Heuristic")
	print("-------------------------------------------\n")

	if(to_print.lower() == "yes"):
		print("-------------------------------------------")
		print("Pre-Pruned Tree")
		print("-------------------------------------------")
		print("Printing the learned tree: \n")
		dtree.printDecisionTree(dtree.root)

	print("\n-------------------------------------------")
	print("Pre-Pruned Accuracy")
	print("-------------------------------------------")
	print("Number of training instances = " + str(df.shape[0]))
	print("Number of training attributes = " + str(df.shape[1] -1))
	print("Total number of nodes in the tree = " + str(dtree.countNumOfNodes(dtree.root)))
	print("Number of leaf nodes in the tree = " + str(dtree.countNumOfLeafs(dtree.root)))
	print("Accuracy of the model on the training dataset = " + str(calculateAccuracy(df,dtree)) + "%\n")
	print("Number of validation instances = " + str(dvalidation.shape[0]))
	print("Number of validation attributes = " + str(dvalidation.shape[1]-1))
	print("Accuracy of the model on the validation dataset before pruning = " + str(calculateAccuracy(dvalidation,dtree)) + "%\n")
	print("Number of testing instances = " + str(dtest.shape[0]))
	print("Number of testing attributes = " + str(dtest.shape[1]-1))
	print("Accuracy of the model on the testing dataset = " + str(calculateAccuracy(dtest,dtree))+"%\n")

	(pruned_best_tree_test, pruned_best_accuracy_test) = pruneTreeFunction(bestTree, l_initial, k_initial, dtest)
	if(to_print.lower() == "yes"):
		print("-------------------------------------------")
		print("Post-Pruned Tree")
		print("-------------------------------------------")
		print("Printing the best tree for l = " + str(l_initial) + " and k = " + str(k_initial) + "\n")
		pruned_best_tree_test.printDecisionTree(pruned_best_tree_test.root)
		print("\nTest data accuracy after pruning with l = " + str(l_initial) + " and k = " + str(k_initial) + " : " + str(pruned_best_accuracy_test) + "\n")

	print("-------------------------------------------")
	print("Post-Pruned Accuracy")
	print("-------------------------------------------")

	#Loop over 10 combinations of l and k values from l_array and k_array respectively.
	for l, k in  zip(l_array, k_array):
		(pruned_best_tree_test, pruned_best_accuracy_test) = pruneTreeFunction(bestTree, l, k, dtest)
		print("Test data accuracy after pruning with l = " + str(l) + " and k = " + str(k) + " : " + str(pruned_best_accuracy_test))
		origAccuracy = calculateAccuracy(dtest, dtree)
		if(pruned_best_accuracy_test > origAccuracy):
			print("Successfully Pruned with improvement in Accuracy on test data set.\n")
		else:
			print("Pruned but Accuracy did not improve.\n")

	print("-------------------------------------")
	print("Completed Information Gain Heuristic.")
	print("-------------------------------------\n")
	print("Please wait to complete building the Decision Tree using Variance Impurity Heuristic!\n")
	
	#Build a decision tree using Variance Impurity Heuristic.
	dtree_VI = Tree(df)
	dtree_VI.createDecisionTree(df, dtree_VI.root, heuristic=calcVarianceImpurity)
	#Calculate accuracy of the decision tree on validation data set.
	maxAccuracy_VI = calculateAccuracy(dvalidation, dtree_VI)
	#Save it as the best decision tree.
	bestTree_VI = copy.deepcopy(dtree_VI)
	#Count the number of nodes in the best decision tree.
	countOfNodes_VI = bestTree_VI.countNumOfNodes(bestTree_VI.root)

	print("--------------------------------------------")
	print("Build Tree using Variance Impurity Heuristic")
	print("--------------------------------------------\n")

	if(to_print.lower() == "yes"):
            print("--------------------------------------------")
            print("Pre-Pruned Tree")
            print("--------------------------------------------")
            print("Printing the learned tree: \n")
            dtree_VI.printDecisionTree(dtree_VI.root)

	print("\n--------------------------------------------")
	print("Pre-Pruned Accuracy")
	print("--------------------------------------------")
	print("Number of training instances = " + str(df.shape[0]))
	print("Number of training attributes = " + str(df.shape[1] -1))
	print("Total number of nodes in the tree = " + str(dtree.countNumOfNodes(dtree_VI.root)))
	print("Number of leaf nodes in the tree = " + str(dtree_VI.countNumOfLeafs(dtree_VI.root)))
	print("Accuracy of the model on the training dataset = " + str(calculateAccuracy(df,dtree_VI)) + "%\n")
	print("Number of validation instances = " + str(dvalidation.shape[0]))
	print("Number of validation attributes = " + str(dvalidation.shape[1]-1))
	print("Accuracy of the model on the validation dataset before pruning = " + str(calculateAccuracy(dvalidation,dtree_VI)) + "%\n")
	print("")
	print("Number of testing instances = " + str(dtest.shape[0]))
	print("Number of testing attributes = " + str(dtest.shape[1]-1))
	print("Accuracy of the model on the testing dataset = " + str(calculateAccuracy(dtest,dtree_VI)) + "%\n")

	(pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = pruneTreeFunction(bestTree_VI, l_initial, k_initial, dtest)
	if(to_print.lower() == "yes"):
            print("--------------------------------------------")
            print("Post-Pruned Tree")
            print("--------------------------------------------")
            print("Printing the best tree for l = " + str(l_initial) + " and k = " + str(k_initial) + "\n")
            pruned_best_tree_test.printDecisionTree(pruned_best_tree_test.root)
            print("\nTest data accuracy after pruning with l = " + str(l_initial) + " and k = " + str(k_initial) + " : " + str(pruned_best_accuracy_test_VI) + "\n")

	print("--------------------------------------------")
	print("Post-Pruned Accuracy")
	print("--------------------------------------------")

	#Loop over 10 combinations of l and k values from l_array and k_array respectively.
	for l, k in  zip(l_array, k_array):
            (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = pruneTreeFunction(bestTree_VI, l, k, dtest)
            print("Test data accuracy after pruning with l = " + str(l) + " and k = " + str(k) + " : " + str(pruned_best_accuracy_test_VI))
            origAccuracy = calculateAccuracy(dtest, dtree_VI)
            if(pruned_best_accuracy_test_VI > origAccuracy):
		print("Successfully Pruned with improvement in Accuracy on test data set.\n")
            else:
		print("Pruned but Accuracy did not improve.\n")

	print("--------------------------------------")
	print("Completed Variance Impurity Heuristic.")
	print("--------------------------------------\n")
	
	end = time.time()
	elapsed = end - start
	print("Current simulation took %f seconds to complete." % elapsed)

	print("\nThank You ! Program ran Successfully.")

if __name__ == '__main__':
    main()
	
#To save the stdout (terminal print statements output) directly to a file, uncomment below line.
#sys.stdout.close()
