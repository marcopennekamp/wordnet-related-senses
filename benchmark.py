import numpy as np
from nltk.corpus import wordnet as wn

import relatedness
import clustering


class Ideal:
    def __init__(self, word, expected):
        self.word = word
        self.expected = expected


def group(*synsets):
    return frozenset(synsets)


ideals = [
    Ideal('cannon', {
        group(wn.synset('cannon.n.03')),
        group(wn.synset('carom.n.02'), wn.synset('cannon.v.01')),
        group(wn.synset('cannon.n.02'), wn.synset('cannon.n.04'), wn.synset('cannon.n.01'),
              wn.synset('cannon.v.02')),
        group(wn.synset('cannon.n.05'))
    })
]


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


def benchmark():
    for word_result in ideals:
        benchmark_ideal(word_result)


benchmark()
