import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.tree as tree
import sklearn.model_selection as model_select
import sklearn.metrics as metrics

def missingValues(df):
    '''
    Takes the dataset and outputs a table of specific information

    Args: df -- Dataset to find values of

    Returns: Table -- Dictionary of values
    '''

    Table = {}

    M = df.shape[0]
    Table["number of instances"] = M

    nullInstances = df.isnull() # location of null instances
    totalMissing = nullInstances.sum().sum() # sum counts for each column
    Table["number of missing values"] = totalMissing 
    Table["fraction of values missing"] = totalMissing/(df.size-M)

    nullRows = M - df.dropna().shape[0]
    Table["num of inst's missing values"] = nullRows
    Table["fraction of inst's missing values"] = nullRows/M
    return Table

def discreteClasses(df):
    '''
    Converts a dataframe to discrete values using sklearn LabelEnconder 
    Converts NaN to missing 

    Args: df -- Pandas dataframe

    Returns: discClasses -- Dataframe of discrete classes'''

    df.fillna("missing", inplace = True)

    discClasses = {}

    for col in df:
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        discClasses[col] = le

    return discClasses


def encodeLabels(df, discClasses):
    '''
    Encodes a dataframe using discrete label template

    Args: df -- Dataframe to be converted
        discClasses -- template of discrete classes to use

    Returns: copy -- Dataframe of discrete classes
    '''

    df.fillna("missing", inplace = True)

    copy = df.copy(deep = True)

    for col in copy:
        copy[col] = discClasses[col].transform(copy[col])
    return copy


def trainTestSplit(df):
    '''
    Splits dataframe to train and test sets.

    Args: df -- Dataframe to split
    
    Returns: Xtrain, Xtest, Ytrain, Ytest 
            -- X (features) and Y (labels) train and test sets 
    '''

    Xtrain = df.iloc[:,:-1]
    Ytrain = df["class"]
    Xtrain, Xtest, Ytrain, Ytest = model_select.train_test_split(Xtrain, Ytrain, random_state=0 )
    return Xtrain, Xtest, Ytrain, Ytest

def trainModel(Xtrain, Ytrain):
    '''
    Trains a decision tree classifier 

    Args: Xtrain, Ytrain -- Training features and labels to train on

    Returns: clf -- trained model 
    '''

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(Xtrain,Ytrain)
    return clf
    
def testModel(model,Xtest,Ytest):
    ''' 
    Tests model on input data and returns error rate
    
    Args: model -- model to test
        Xtest, Ytest -- test data and labels
    
    Returns: errorRate -- rate of error of model
    '''

    testScore = model.score(Xtest, Ytest)
    errorRate = 1-testScore
    return errorRate 


def splitDataSet(df):
    '''
    Creates dataset comprised of 50% examples missing values
    and 50% without missing values from the input dataset

    Args: df -- Dataset to manipulate

    Returns: halfNan -- Dataset with 50% of examples containing at least one NaN
    '''
    noNan = df.dropna()
    allNan = df[df.isna().any(axis=1)]

    halfNan = allNan.append(noNan.sample(allNan.shape[0]))

    # Shuffle datapoints
    halfNan = halfNan.sample(frac=1)
    
    return halfNan 


# converts values with 'missing' to 
# most common value in column
def missingToModal(df):
    '''
    Replaces NaN values with modal attribute value

    Args: df -- Dataset

    Returns: dfModal -- Datset with replaced NaN values with modal value
    '''
    dfModal = df.copy(deep = True)
    for col in dfModal:
        dfModal[col].fillna(dfModal[col].mode()[0],inplace=True)
    return dfModal
    


# ****************
# Initialisation
# ****************

adultDataOrig = pd.read_csv("./data/adult.csv", dtype=str).drop("fnlwgt",1)
adultData = adultDataOrig.copy(deep=True)

# ****************
# 1.1
# ****************
missing = missingValues(adultData)
print(missing)

# ****************
# 1.2
# ****************

# get discrete classes
discClasses = discreteClasses(adultData)
# encode df with discrete classes
dfDiscrete = encodeLabels(adultData,discClasses)
# show discrete values
for col in dfDiscrete.columns:
    print("{} : {}".format(col, np.unique(getattr(dfDiscrete,col))))

# ****************
# 1.3
# ****************

dfDropNa = adultData.dropna().copy(deep = True)
dfDropNaEncoded = encodeLabels(dfDropNa, discClasses)
Xtrain, Xtest, Ytrain, Ytest = trainTestSplit(dfDropNaEncoded)
model = trainModel(Xtrain,Ytrain)
errorRate = testModel(model,Xtest,Ytest)
print("Ignored missing attr's error rate = ", errorRate)


# ****************
# 1.4
# ****************

# D   - entire dataset (adultData)
# D'  - dataset with all missing values + same number without missing values
# D'1 - converted NaN to 'missing'
# D'2 - replaced NaN with modal column value

# Create new datasets
D1 = splitDataSet(adultDataOrig) # contains NaN
D2 = missingToModal(D1)

# Encode datasets
D1Encoded = encodeLabels(D1,discClasses)
D2Encoded = encodeLabels(D2,discClasses)

# Train on D1 data and test on D
Xtrain = D1Encoded.iloc[:,:-1]
Ytrain = D1Encoded.iloc[:,-1]
model = trainModel(Xtrain,Ytrain)
errorRate = testModel(model,Xtest,Ytest)
print("D1 error rate = ", errorRate)


# Train on D2 data and test on D
Xtrain = D2Encoded.iloc[:,:-1]
Ytrain = D2Encoded.iloc[:,-1]
model = trainModel(Xtrain,Ytrain)
errorRate = testModel(model,Xtest,Ytest)
print("D2 error rate = ", errorRate)


