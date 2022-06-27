"""
Implements k-means clustering method.
"""

from typing import List

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans


def find_inertias(points: List[np.ndarray], max_clusters: int) -> list:
  """
  Find the inertia value for each k.
  """
  inertia = []

  for k in range(1, max_clusters + 1):

    if k > len(points):
      inertia.append(0)
      continue
    
    km = KMeans(n_clusters=k)
    km = km.fit(points)
    inertia.append(km.inertia_)

  return inertia

def k_means_clustering(points: List[np.ndarray], max_clusters: int) -> List[np.ndarray]:
  """
  Takes a list of points and returns the list of means for each of the cluster

  Key arguments:
    points - the points to apply k-means clustering on.
    max_clusters - the maximum possible number of clusters
  """

  # find the inertials for all ks
  inertias = find_inertias(points, max_clusters)


  # find the ideal number of clusters
  kl = KneeLocator(range(1, max_clusters + 1), inertias,
                   curve='convex', direction='decreasing')
  num_clusters = kl.elbow

  if num_clusters is None:
    num_clusters = 1

  kmeans = KMeans(n_clusters=num_clusters)
  kmeans.fit(points)

  centers = np.array(kmeans.cluster_centers_)
  
  return centers






