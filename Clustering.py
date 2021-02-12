import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import itertools

def table(df):
    tabl = {}
    for attr in df:
        mean = df[attr].mean().round(2)
        rang = [df[attr].min(),df[attr].max()]
        tabl[attr] = [mean,rang]
    return tabl
    # print(tabl)    

def kmeans(df, k=3):
    X = df.values
    # split each x in X into 3 clusters
    clusters = KMeans(n_clusters=k,random_state=0).fit_predict(X)
    # return array of corresponding clusters
    return clusters
    


def plotKmeans(X, clusters):

    pairs = [pair for pair in itertools.combinations(X.columns,2)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = clusters

    n = 0
    for pair in pairs:
        ax = fig.add_subplot(n+1,1,n+1)

        Xsample = X[[ pair[0] , pair[1]]]
        for i in np.unique(clusters):
            c = "Cluster:" + str(i+1)
            plt.scatter(Xsample.iloc[y==i,0],Xsample.iloc[y==i,1], c='red',label=c)
        # plt.scatter(Xsample.iloc[y==0,0],Xsample.iloc[y==0,1], c='red',label='Cluster 1')
        # plt.scatter(Xsample.iloc[y==1,0],Xsample.iloc[y==1,1], c='green',label='Cluster 2')
        # plt.scatter(Xsample.iloc[y==2,0],Xsample.iloc[y==2,1], c='cyan',label='Cluster 3')
        plt.show()

customerData = pd.read_csv('./data/wholesale_customers.csv',dtype=float)

customerData.drop(["Channel","Region"],axis=1, inplace=True)

tabl = table(customerData)
for attr in tabl:
    print(attr, " : ", tabl[attr])

clusters = kmeans(customerData)
plotKmeans(customerData,clusters)