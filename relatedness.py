import metonym
import itertools
import numpy as np
from nltk.corpus import wordnet as wn


def derivationally_related_synsets(lemma):
    return list(map(lambda form: form.synset(), lemma.derivationally_related_forms()))


def derivational_similarity(lemma1, lemma2):
    """
    Calculates the similarity of lemma1 and lemma2 based on the similarity of the best derivationally related
    form that is noted in either lemma1 or lemma2. The 'best' form is defined by taking the maximum of similarities
    between all derivationally related forms of lemma1, and lemma2; and all derivationally related forms of lemma2,
    and lemma1.

    :return: The similarity of lemma1 and lemma2, based on the similarity of the best derivationally related form.
             If there is no such derivationally related form that leads to a valid similarity value, None is returned.
    """

    def get_candidates(related_lemma, lemma):
        return list(itertools.product(derivationally_related_synsets(related_lemma), [lemma.synset()]))

    candidates = get_candidates(lemma1, lemma2) + get_candidates(lemma2, lemma1)
    similarities = list(filter(lambda sim: sim is not None, map(lambda c: c[0].wup_similarity(c[1]), candidates)))
    if similarities:
        return max(similarities)
    else:
        return None


def get_lemma_for_word(synset, word):
    """
    :return: The lemma for the given synset and word. For example, for a synset dog = wn.synset('dog.n.01'), we can
             retrieve the lemma 'dog.n.01.dog' by calling get_lemma_for_word(dog, 'dog').
    """
    return wn.lemma(f'{synset.name()}.{word}')


def compute_naive_similarity_matrix(word, synsets):
    # First build the distance matrix by computing the similarity of the synsets.
    def similarity(synset1, synset2):
        # If the POS of the synsets are equal, we can apply the Wu-Palmer similarity measure.
        # Otherwise, applying Wu-Palmer may give us a None result. In such a case, we fallback
        # to the derivational_similarity measure defined above.
        sim = synset1.wup_similarity(synset2)
        if sim is None:
            sim = derivational_similarity(get_lemma_for_word(synset1, word), get_lemma_for_word(synset2, word))

        # If the similarity is still None (both measures failed), we set it to 0.0.
        if sim is None:
            sim = 0.0

        # print(f'{synset1.definition()} | {synset2.definition()}:\t{sim}')
        return sim

    return np.array([[similarity(s1, s2) for s1 in synsets] for s2 in synsets])


def load_wordnet_graph(graph_name):
    return metonym.WordNetGraph(f'{graph_name}.gml')


def compute_lch_similarity_matrix(wordnet_graph, nodes):
    return np.array([[wordnet_graph.lch_similarity(n1, n2) for n1 in nodes] for n2 in nodes])
