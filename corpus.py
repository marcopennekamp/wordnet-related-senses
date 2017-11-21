from nltk.corpus import wordnet as wn


def synsets(word):
    """
    Fetches the synsets for a word like wn.synsets, but doesn't return synsets that are morphologically related to
    the word. For example, when fetching synsets for the word 'deserted', only synsets that include 'deserted' exactly
    will be returned, not synsets for verbs like 'desert' (like wn.synsets would return).
    """

    # Just filter all synsets that don't include the word as is.
    return list(filter(lambda synset: word in synset.lemma_names(), wn.synsets(word)))
