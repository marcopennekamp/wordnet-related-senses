import sklearn.cluster
import numpy as np


# Cluster with affinity propagation (doesn't require the number of clusters).
def cluster_affinity(word, nodes, similarity_matrix):
    """
    Cluster with affinity propagation (doesn't require the number of clusters).

    :return: A set of clusters.
    """
    aff = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    aff.fit(similarity_matrix)
    clusters = set()
    for cluster_id in np.unique(aff.labels_):
        nodes_in_cluster = nodes[np.nonzero(aff.labels_ == cluster_id)]
        synsets = map(lambda node: node.synset, nodes_in_cluster)
        cluster = frozenset(synsets)
        clusters.add(cluster)
    return clusters
