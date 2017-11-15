import numpy as np
from sys import argv
from nltk.corpus import wordnet as wn

import relatedness
import clustering


def main():
    _, word = argv
    synsets = np.asarray(wn.synsets(word))

    similarity_matrix = relatedness.compute_metonym_similarity_matrix(word, synsets)
    clusters = clustering.cluster_affinity(word, synsets, similarity_matrix)

    print(f'Clusters for \'{word}\':')
    for cluster in clusters:
        print(f' - {next(iter(cluster)).name()}: {cluster}')


# Call the main function defined above to execute the program.
main()
