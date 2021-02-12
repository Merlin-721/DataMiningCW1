import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
from matplotlib import cm

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
    fitter = KMeans(n_clusters=k,random_state=0).fit(X)
    clusters = fitter.predict(X)
    # return array of corresponding clusters
    return fitter, clusters
    


def plotKmeans(X, clusters, k):

    pairs = [pair for pair in itertools.combinations(X.columns,2)]

    for pair in pairs:

        colours = iter(cm.rainbow(np.linspace(0, 1, k)))

        Xsample = X[[ pair[0] , pair[1]]]
        for i in np.unique(clusters):
            c = "Cluster:" + str(i+1)
            plt.scatter(Xsample.iloc[clusters==i,0],Xsample.iloc[clusters==i,1], color=next(colours),label=c)

        name = (pair[0] + " " + pair[1] + ".png")
        plt.savefig(name)
        plt.clf()


# INITIALISATION

customerData = pd.read_csv('./data/wholesale_customers.csv',dtype=float)

customerData.drop(["Channel","Region"],axis=1, inplace=True)


# Part 1
tabl = table(customerData)
for attr in tabl:
    print(attr, " : ", tabl[attr])


# Part 2
k=3
fitter, clusters = kmeans(customerData,k)
plotKmeans(customerData,clusters,k)

# Part 3
kSet = [3,5,10]

for i in kSet:
    fitter, clusters = kmeans(customerData,i)
    BC = euclidean_distances(fitter.cluster_centers_) 
    WC
    ratio = WC/BC