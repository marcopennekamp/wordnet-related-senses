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
