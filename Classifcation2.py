import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.tree as tree
import sklearn.model_selection as model_select
import sklearn.metrics as metrics

def missingValues(df):

    M = df.shape[0]

    Table = {}

    Table["number of instances"] = M
    nullInstances = df.isnull() # location of null instances
    totalMissing = nullInstances.sum().sum() # sum counts for each column
    Table["number of missing values"] = totalMissing 
    Table["fraction missing values"] = totalMissing/(df.size-M)

    nullRows = M - df.dropna().shape[0]
    Table["num of inst's missing values"] = nullRows
    Table["fraction of inst's missing values"] = nullRows/M
    return Table

# converts NaN to missing
# and converts values to discrete values
def discreteClasses(df):
    df.fillna("missing", inplace = True)
    discClasses = {}
    for col in df:
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        discClasses[col] = le
    return discClasses


def encodeLabels(df, discClasses):
    df.fillna("missing", inplace = True)
    copy = df.copy(deep = True)
    for col in copy:
        copy[col] = discClasses[col].transform(copy[col])
    return copy


def trainTestModel(df):

    X = df.iloc[:,:-1]
    Y = df["class"]
    X_train, X_test, y_train, y_test = model_select.train_test_split(X, Y, random_state=0 )

    # initialise the decision tree
    clf = tree.DecisionTreeClassifier( random_state = 0 )
    # fit the tree model to the training data
    clf.fit(X_train, y_train)

    testScore = clf.score(X_test, y_test)
    errorRate = 1-testScore

    return errorRate 


# splits dataset into 3 sets:
# instances with NaN removed
# only instances with NaN
# 50/50 NaN/non-NaN
def splitDataSet(df):
    # noNan  = df[~df.isin(["missing"]).any(axis=1)]
    # allNan = df[df.isin(["missing"]).any(axis=1)]
    noNan = df.dropna()
    allNan = df[df.isna().any(axis=1)]

    halfNan = allNan.append(noNan.sample(allNan.shape[0]))
    return halfNan 


# converts values with 'missing' to 
# most common value in column
def missingToModal(df):
    dfModal = df.copy(deep = True)
    for col in dfModal:
        dfModal[col].fillna(dfModal[col].mode()[0],inplace=True)
        # column = dfModal[col]
        # vals,counts = np.unique(column[~np.isnan(column)],return_counts=True)
        # vals = np.unique(column[~np.isna(column)])
        # index = np.argmax(counts)
        # dfModal[col].fillna(vals[index])
        # np.where(dfModal[col] != 'missing',dfModal[col], vals[index])
    return dfModal
    


adultDataOrig = pd.read_csv("./data/adult.csv", dtype=str).drop("fnlwgt",1)

adultData = adultDataOrig.copy(deep=True)

#****************
# Part 1
# ****************
missing = missingValues(adultData)
print(missing)

#****************
# Part 2
# ****************

# get discrete classes
discClasses = discreteClasses(adultData)
# encode df with discrete classes
dfDiscrete = encodeLabels(adultData,discClasses)
# show discrete values
for col in dfDiscrete.columns:
    print("{} : {}".format(col, np.unique(getattr(dfDiscrete,col))))

#****************
# Part 3
#****************

errorRate = trainTestModel(dfDiscrete)
print("error rate = ", errorRate)

#****************
# Part 4
# ****************

# D - entire dataset (adultData)

# D' - dataset with all missing values + same number without missing values

# D'1 - convert NaN to 'missing'

# D'2 - replace NaN with modal column value

D1 = splitDataSet(adultDataOrig) # contains NaN
# D1.fillna("missing", inplace = True)

D2 = missingToModal(D1)

D1Encoded = encodeLabels(D1,discClasses)
D2Encoded = encodeLabels(D2,discClasses)

errorD1 = trainTestModel(D1Encoded)
print(errorD1)
errorD2 = trainTestModel(D2Encoded)
print(errorD2)

