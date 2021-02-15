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
        colours = iter(cm.get_cmap("copper")(np.linspace(0, 1, k)))

        Xsample = X[[ pair[0] , pair[1]]]

        for i in np.unique(clusters):
            c = "Cluster:" + str(i+1)
            plt.scatter(Xsample.iloc[clusters==i,0],Xsample.iloc[clusters==i,1], color=next(colours),label=c)

        name = (pair[0] + " " + pair[1] + ".png")
        plt.savefig(name)
        plt.clf()

def betweenClusterScore(k,fitter):
    bc = euclidean_distances(fitter.cluster_centers_)
    dists = 0
    for c in bc:
        # square distances
        c = c[list(c).index(0):]
        # sum the distances
        dists += sum([n**2 for n in c])
    return dists


def withinClusterScore(X,clusterLabels,fitter, k):
    wc = []
    for i in range(k):
        dists = 0
        # get feature vectors
        clustFeats = X.loc[clusterLabels == i] 
        for row in clustFeats.iterrows():
            dist = np.linalg.norm(fitter.cluster_centers_[i] - row[1].values)
            dists += dist**2
        wc.append(dists)
    return dists

# INITIALISATION

customerData = pd.read_csv('./data/wholesale_customers.csv',dtype=float)

customerData.drop(["Channel","Region"],axis=1, inplace=True)


# Part 1
tabl = table(customerData)
for attr in tabl:
    print(attr, " : ", tabl[attr])


# Part 2
k=3
# fitter, clusters = kmeans(customerData,k)
# plotKmeans(customerData,clusters,k)

# Part 3
kSet = [3,5,10]
BCs,WCs,ratios = [],[],[]
for i in kSet:
    fitter, clusterLabels = kmeans(customerData,i)
    BC = betweenClusterScore(i,fitter)
    WC = withinClusterScore(customerData,clusterLabels,fitter,i)
    BCs.append(BC)
    WCs.append(WC)
    ratios.append(BC/WC)

print(BCs)
print(WCs)
print(ratios)

    # WC
    # ratio = WC/BC