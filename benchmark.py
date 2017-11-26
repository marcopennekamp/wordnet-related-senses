import time
import numpy as np
from munkres import Munkres
from sys import argv

import benchmark_words
import corpus
import relatedness
import clustering
import utils


def calculate_clustering_accuracy(ideal, clusters):
    """
    Calculates the clustering accuracy for a word given the expected (ideal) clusters. The accuracy
    is calculated by assigning each ideal cluster to its best-fit real cluster (given the real cluster
    isn't assigned to another ideal cluster) and then counting the misplaced senses.

    :return: A score from 1.0 (perfect match) to 0.0 (nothing fits). However, 0.0 is theoretically not possible,
             because we don't label the clusters, and thus at least one ideal cluster will be assigned to some
             real cluster containing at least one correct sense. A perfect match is only possible if the number
             of ideal and real clusters is identical.
    """

    # Convert to lists so we can index them.
    ideal_list = list(ideal.expected)
    cluster_list = list(clusters)

    # We want to assign exactly one ideal cluster to one real cluster, so that we can properly count the
    # senses that were misplaced. Additionally, the number of misplaced senses has to be minimised so that
    # we can guarantee the best possible assignment for measuring accuracy.
    # This is a classic assignment problem, which can be solved with the Hungarian algorithm. In this case,
    # for an ideal cluster, the cost is the number of senses that are missing from a given cluster or that
    # are misplaced in the given cluster.
    munkres = Munkres()
    cost_matrix = [[len((ideal_cluster | cluster) - (ideal_cluster & cluster)) for cluster in cluster_list]
                   for ideal_cluster in ideal_list]
    indexes = munkres.compute(cost_matrix)

    all_synset_count = sum(map(lambda cluster: len(cluster), ideal_list))
    correct_synset_count = 0
    for ideal_index, cluster_index in indexes:
        ideal_cluster = ideal_list[ideal_index]
        cluster = cluster_list[cluster_index]
        correct_synset_count += len(ideal_cluster & cluster)
    return correct_synset_count / all_synset_count


def print_benchmark_ideal_results(ideal, clusters, accuracy, verbose):
    word = ideal.word

    if verbose:
        print(f'Expected clusters for \'{word}\':')
        utils.print_clusters(ideal.expected)

        print(f'Clusters for \'{word}\':')
        utils.print_clusters(clusters)

    print(f'Clustering accuracy for \'{word}\': {accuracy}')
    print()


def benchmark_ideal(wordnet_graph, ideal, verbose):
    """
    :return: The accuracy of the clustering for the given word.
    """
    word = ideal.word
    synsets = np.asarray(corpus.synsets(word))

    similarity_matrix = relatedness.compute_lch_similarity_matrix(wordnet_graph, word, synsets)
    clusters = clustering.cluster_affinity(word, synsets, similarity_matrix)
    accuracy = calculate_clustering_accuracy(ideal, clusters)

    print_benchmark_ideal_results(ideal, clusters, accuracy, verbose)

    return accuracy


def benchmark():
    _, graph_name = argv
    verbose = True
    absolute_accuracy = 0
    wordnet_graph = relatedness.load_wordnet_graph(graph_name)
    for word_result in benchmark_words.ideals:
        absolute_accuracy += benchmark_ideal(wordnet_graph, word_result, verbose)
    total_accuracy = absolute_accuracy / len(benchmark_words.ideals)
    print(f'The total clustering accuracy is: {total_accuracy}')


start_time = time.time()
benchmark()
print(f"--- {time.time() - start_time} seconds ---")
