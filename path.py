from sys import argv
from nltk.corpus import wordnet as wn

import relatedness

_, graph_name, synset1_name, synset2_name = argv
synset1 = wn.synset(synset1_name)
synset2 = wn.synset(synset2_name)
wordnet_graph = relatedness.load_wordnet_graph(graph_name)
print(f'path: {wordnet_graph.shortest_path(synset1, synset2)}')
print(f'length: {wordnet_graph.shortest_path_distance(synset1, synset2)}')
