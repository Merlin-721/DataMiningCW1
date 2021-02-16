import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import calinski_harabasz_score

import matplotlib.pyplot as plt
from matplotlib import cm

import itertools


def table(df):
    '''
    Creates table of information about attributes and outputs
    in the format: [mean,[min value, max value]]

    Args: df -- Dataset

    Returns: tabl -- Dictionary of values
    '''

    tabl = {}
    for attr in df:
        mean = df[attr].mean().round(2)
        rang = [df[attr].min(),df[attr].max()]
        tabl[attr] = [mean,rang]
    return tabl

def kmeans(df, k=3):
    '''
    Creates K-Means clustering model and returns
    instances with their respective cluster label

    Args: df -- Dataset to cluster
          k -- Number of clusters

    Returns: fitter -- fitted clustering model
             clusters -- cluster labels of each instance
    '''

    X = df.values
    # split each x in X into 3 clusters
    fitter = KMeans(n_clusters=k,random_state=0).fit(X)
    clusters = fitter.predict(X)
    # return array of corresponding clusters
    return fitter, clusters
    


def plotKmeans(X, clusters, k):
    '''
    Creates PNGs of pairwise cluster combinations

    Args: X -- Data features
          clusters -- Instance cluster labels
          k -- Number of clusters

    '''
    pairs = [pair for pair in itertools.combinations(X.columns,2)]

    
    for pair in pairs:
        colours = iter(cm.get_cmap("copper")(np.linspace(0, 1, k)))

        Xsample = X[[ pair[0] , pair[1]]]

        for i in np.unique(clusters):
            c = "Cluster:" + str(i+1)
            plt.scatter(Xsample.iloc[clusters==i,0],Xsample.iloc[clusters==i,1], color=next(colours),label=c)
        plt.suptitle(str(pair[0] + " vs " + pair[1] + ", k = " + str(k)),fontsize=20)
        plt.xlabel(pair[0], fontsize=15)
        plt.ylabel(pair[1], fontsize=15)
        name = (pair[0] + " " + pair[1] + ".png")
        plt.savefig(name)
        plt.clf()

def betweenClusterScore(fitter):
    '''
    Calculates between cluster score of model
    i.e., between cluster centres

    Args: fitter -- Trained k-means model

    Returns: dists -- Array of pairwise distances from each cluster to one another
    '''
    # Get Euclid distances between cluster centres
    bc = euclidean_distances(fitter.cluster_centers_)
    dists = 0
    for c in bc:
        # Filter out cluster dist to itself
        c = c[list(c).index(0):]
        # Sum the squared distances
        dists += sum([n**2 for n in c])
    return dists

# def calinskiHarabaz(ratio,k,clusterLabels):
#     n = len(clusterLabels)
#     return ratio * (n-k)/(k-1)


# ****************
# Clustering Initialisation
# ****************

customerData = pd.read_csv('./data/wholesale_customers.csv',dtype=float)
customerData.drop(["Channel","Region"],axis=1, inplace=True)


# ****************
# 2.1
# ****************

tabl = table(customerData)
for attr in tabl:
    print(attr, " : ", tabl[attr])


# ****************
# 2.2
# ****************

k=3
# fitter, clusters = kmeans(customerData,k)
# plotKmeans(customerData,clusters,k)


# ****************
# 2.3
# ****************
kSet = [3,5,10]
BCs,WCs,ratios,caliHs = [],[],[],[]
for i in kSet:
    fitter, clusterLabels = kmeans(customerData,i)
    BC = betweenClusterScore(fitter)
    WC = fitter.inertia_
    ratio = BC/WC
    # caliH = calinskiHarabaz(ratio,k,clusterLabels)
    caliH = calinski_harabasz_score(customerData,clusterLabels)
    BCs.append(BC)
    WCs.append(WC)
    ratios.append(ratio)
    caliHs.append(caliH)

print("BC Scores: " + str(BCs))
print("WC Scores: " + str(WCs))
print("Ratios: " + str(ratios))
print("Calinksi-Harabaz's: " + str(caliHs))
