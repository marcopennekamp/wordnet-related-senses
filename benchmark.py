import numpy as np

import benchmark_words
import corpus
import relatedness
import clustering
import utils


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

    if verbose:
        print(f'Expected clusters for \'{word}\':')
        utils.print_clusters(ideal.expected)

        print(f'Clusters for \'{word}\':')
        utils.print_clusters(clusters)

    print(f'Clustering accuracy for \'{word}\': {accuracy}')


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
    for word_result in benchmark_words.ideals:
        absolute_accuracy += benchmark_ideal(word_result, verbose)
    total_accuracy = absolute_accuracy / len(benchmark_words.ideals)
    print(f'The total clustering accuracy is: {total_accuracy}')


benchmark()
