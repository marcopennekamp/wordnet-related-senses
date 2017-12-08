from sys import argv
import metonym

import relatedness

_, graph_name, node1_key, node2_key = argv
node1 = metonym.Node.from_key(node1_key)
node2 = metonym.Node.from_key(node2_key)
wordnet_graph = relatedness.load_wordnet_graph(graph_name)
print(f'path: {wordnet_graph.shortest_path(node1, node2)}')
print(f'length: {wordnet_graph.shortest_path_distance(node1, node2)}')
