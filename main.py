import numpy as np
from sys import argv

import corpus
import relatedness
import clustering


def main():
    _, word = argv
    synsets = np.asarray(corpus.synsets(word))

    if len(synsets) > 1:
        similarity_matrix = relatedness.compute_lch_similarity_matrix(word, synsets)
        clusters = clustering.cluster_affinity(word, synsets, similarity_matrix)

        def cluster_to_group_string(cluster):
            names = list(map(lambda synset: f"'{synset.name()}'", cluster))
            return f'group({", ".join(names)})'

        print(f'Clusters for \'{word}\':')
        for cluster in clusters:
            print(f' - {cluster_to_group_string(cluster)}')
    else:
        print(f'There is no need to cluster the senses of {word}, because it only has one.')


# Call the main function defined above to execute the program.
main()
