import numpy as np
from sys import argv

import corpus
import relatedness
import clustering
import utils


def main():
    _, graph_name, word = argv
    nodes = np.asarray(corpus.nodes(word))
    wordnet_graph = relatedness.load_wordnet_graph(graph_name)

    if len(nodes) > 1:
        similarity_matrix = relatedness.compute_lch_similarity_matrix(wordnet_graph, nodes)
        clusters = clustering.cluster_affinity(word, nodes, similarity_matrix)
        print(f'Clusters for \'{word}\':')
        utils.print_clusters(clusters)
    elif len(nodes) == 1:
        print(f'There is no need to cluster the senses of {word}, because it only has one.')
    else:
        print(f'The word {word} does not exist.')


# Call the main function defined above to execute the program.
main()
