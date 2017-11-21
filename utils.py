

def cluster_to_string(cluster):
    names = list(map(lambda synset: f"'{synset.name()}'", cluster))
    return ", ".join(names)


def print_clusters(clusters):
    for cluster in clusters:
        print(f' - {cluster_to_string(cluster)}')
