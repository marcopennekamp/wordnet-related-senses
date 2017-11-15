import numpy as np
from nltk.corpus import wordnet as wn

import relatedness
import clustering


class Ideal:
    def __init__(self, word, expected):
        self.word = word
        self.expected = expected


def group(*synset_names):
    synsets = map(lambda name: wn.synset(name), synset_names)
    return frozenset(synsets)


ideals = [
    Ideal('cannon', {
        group('cannon.n.01', 'cannon.n.02', 'cannon.n.04', 'cannon.v.02'),
        group('carom.n.02', 'cannon.v.01'),
        group('cannon.n.03'),
        group('cannon.n.05')
    }),
    Ideal('model', {
        group('model.n.01', 'model.n.04', 'model.n.07', 'model.n.09', 'model.v.01', 'model.v.05', 'model.v.06'),
        group('model.n.02'),
        group('mannequin.n.01', 'model.n.03', 'model.v.02', 'model.v.03', 'model.v.04'),
        group('exemplar.n.01', 'model.n.06', 'exemplary.s.01'),
    })
]


def measure_clustering_quality(ideal, clusters):
    """

    :return: A score from 1.0 (perfect match) to 0.0 (nothing fits). However, 0.0 is theoretically not possible,
             because we don't label the clusters, and thus at least one ideal cluster will find a single element
             as a counterpart.
    """
    absolute_score = 0.0
    for ideal_cluster in ideal.expected:
        # Implements the Jaccard similarity measure.
        def measure_cluster_similarity(cluster):
            return len(cluster & ideal_cluster) / len(cluster | ideal_cluster)

        similarities = list(map(measure_cluster_similarity, clusters))
        # TODO: This gives the same weight to all clusters. We should weigh each cluster by its size.
        absolute_score += max(similarities)
    return absolute_score / len(ideal.expected)


def benchmark_ideal(ideal):
    word = ideal.word
    synsets = np.asarray(wn.synsets(word))

    similarity_matrix = relatedness.compute_metonym_similarity_matrix(word, synsets)
    clusters = clustering.cluster_affinity(word, synsets, similarity_matrix)

    def print_cluster(c):
        print(f' - {next(iter(c)).name()}: {c}')

    print(f'Expected clusters for \'{word}\':')
    for cluster in ideal.expected:
        print_cluster(cluster)

    print(f'Clusters for \'{word}\':')
    for cluster in clusters:
        print_cluster(cluster)

    print(f'Clustering quality: {measure_clustering_quality(ideal, clusters)}')


def benchmark():
    for word_result in ideals:
        benchmark_ideal(word_result)


benchmark()
