import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


DATA_DIR = 'data/'
DATA_FILE = 'adult.csv'

f = open(DATA_DIR+DATA_FILE,'rt')

rawdata0 = csv.reader( f )
rawdata = [rec for rec in rawdata0] # rawdata is array of each row in string form

# Y = []
# numAttrs = len(rawdata[0])

# remove fnlwgt
for row in rawdata:
    del row[2]
    # Y.append(row[numAttrs-1])
    # del row[numAttrs-1]

# remove headers 
header = rawdata[0]
del rawdata[0]
# del Y[0]

# Part 1 
# *******************
M = len(rawdata)

numMissingVals = 0 # total number of missing values
numInstMissingVals = 0 # instances that are missing values


for row in rawdata:
    if '' in row:
        numInstMissingVals+=1
        numMissingVals += row.count('')

fractionMissingVals = numMissingVals/(M*len(row)) # fraction of all values missing

fractionMissingInst = numInstMissingVals/M # fraction of instances with missing values

# remove the print from this
tableData = [M,numMissingVals,fractionMissingVals,numInstMissingVals,fractionMissingInst]
print(tableData)


# Part 2 
# *******************
dataNew = np.transpose(np.array(rawdata)) # dataNew is array of columns

df = pd.DataFrame(dict(zip(header,dataNew))) # df is dataframe of columns including class

df = df.replace('', np.nan)

# Create df with no missing vals
dfWithoutMissing = df.dropna()

dfDiscrete = df.apply(preprocessing.LabelEncoder().fit_transform)

for col in dfDiscrete.columns:
    print("{} : {}".format(col, np.unique(getattr(dfDiscrete,col))))



# Part 3
# *******************

import sklearn.tree as tree
import sklearn.model_selection as model_select
import sklearn.metrics as metrics

# Y = dfDiscrete["class"]
# del dfDiscrete["class"]

Xtrain, Xtest, Ytrain, Ytest = model_select.train_test_split( dfDiscrete.values, Y, random_state=0 )
M_train = len( Xtrain )
M_test = len( Xtest )
# print('number of training instances = ' + str( M_train ))
# print('number of test instances = ' + str( M_test ))

# initialise the decision tree
clf = tree.DecisionTreeClassifier( random_state = 0 )

# fit the tree model to the training data
clf.fit( Xtrain, Ytrain )

# predict the labels for the test set
y_hat = clf.predict( Xtest )


# count the number of correctly predicted labels
# count = 0.0
# for i in range( M_test ):
#     if ( y_hat[i] == np.array(Ytest)[i] ):
#         count += 1
# score = ( count / M_test ) # score is proportion of correct predictions

trainScore = clf.score(Xtrain, Ytrain)
testScore = clf.score(Xtest, Ytest)
accuracy = metrics.accuracy_score(Ytest,y_hat)

errorRate = 1-testScore
# print('training score = ', trainScore)
print('test score = ', testScore)
print("error rate = ", errorRate)


# Part 4 
# ****************


