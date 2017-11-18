import numpy as np
from nltk.corpus import wordnet as wn

import corpus
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
        group('cannon.n.05'),
    }),
    Ideal('model', {
        group('model.n.01', 'model.n.04', 'model.n.07', 'model.n.09', 'model.v.01', 'model.v.05', 'model.v.06'),
        group('model.n.02'),
        group('mannequin.n.01', 'model.n.03', 'model.v.02', 'model.v.03', 'model.v.04'),
        group('exemplar.n.01', 'model.n.06', 'exemplary.s.01'),
    }),
    Ideal('dog', {
        group('cad.n.01', 'dog.n.03', 'frump.n.01'),
        group('pawl.n.01', 'andiron.n.01'),
        group('dog.n.01', 'chase.v.01'),
        group('frank.n.02'),
    }),
    Ideal('arrange', {
        group('arrange.v.01', 'arrange.v.07'),
        group('arrange.v.02', 'stage.v.02'),
        group('arrange.v.06'),
        group('dress.v.16'),
        group('format.v.01'),
    }),
]


def measure_clustering_accuracy(ideal, clusters):
    """

    :return: A score from 1.0 (perfect match) to 0.0 (nothing fits). However, 0.0 is theoretically not possible,
             because we don't label the clusters, and thus at least one ideal cluster will find a single element
             as a counterpart.
    """
    absolute_score = 0.0
    synset_count = 0
    for ideal_cluster in ideal.expected:
        # Implements the Jaccard similarity measure.
        def measure_cluster_similarity(cluster):
            return len(cluster & ideal_cluster) / len(cluster | ideal_cluster)

        similarities = list(map(measure_cluster_similarity, clusters))

        # To give each sense the same weight, instead of each cluster, we multiply the score by the number of synsets
        # in the ideal cluster. At the end, we also divide by the number of unique synsets in all ideal clusters.
        current_synset_count = len(ideal_cluster)
        synset_count += current_synset_count
        absolute_score += max(similarities) * current_synset_count
    return absolute_score / synset_count


def print_benchmark_ideal_results(ideal, clusters, accuracy, verbose):
    word = ideal.word

    def print_cluster(c):
        print(f' - {next(iter(c)).name()}: {c}')

    if verbose:
        print(f'Expected clusters for \'{word}\':')
        for cluster in ideal.expected:
            print_cluster(cluster)

        print(f'Clusters for \'{word}\':')
        for cluster in clusters:
            print_cluster(cluster)

    print(f'Clustering accuracy for \'{word}\': {measure_clustering_accuracy(ideal, clusters)}')


def benchmark_ideal(ideal, verbose):
    """
    :return: The accuracy of the clustering for the given word.
    """
    word = ideal.word
    synsets = np.asarray(corpus.synsets(word))

    similarity_matrix = relatedness.compute_lch_similarity_matrix(word, synsets)
    clusters = clustering.cluster_affinity(word, synsets, similarity_matrix)
    accuracy = measure_clustering_accuracy(ideal, clusters)

    print_benchmark_ideal_results(ideal, clusters, accuracy, verbose)

    return accuracy


def benchmark():
    verbose = True
    absolute_accuracy = 0
    for word_result in ideals:
        absolute_accuracy += benchmark_ideal(word_result, verbose)
    total_accuracy = absolute_accuracy / len(ideals)
    print(f'The total clustering accuracy is: {total_accuracy}')


benchmark()
