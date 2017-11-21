import numpy as np
from sys import argv

import corpus
import relatedness
import clustering
import utils


def main():
    _, word = argv
    synsets = np.asarray(corpus.synsets(word))
    wordnet_graph = relatedness.load_wordnet_graph()

    if len(synsets) > 1:
        similarity_matrix = relatedness.compute_lch_similarity_matrix(wordnet_graph, word, synsets)
        clusters = clustering.cluster_affinity(word, synsets, similarity_matrix)
        print(f'Clusters for \'{word}\':')
        utils.print_clusters(clusters)
    elif len(synsets) == 1:
        print(f'There is no need to cluster the senses of {word}, because it only has one.')
    else:
        print(f'The word {word} does not exist.')


# Call the main function defined above to execute the program.
main()
