import itertools
from nltk.corpus import wordnet as wn


def main():
    synsets = wn.synsets('cannon')
    print(synsets)

    pairs = itertools.combinations(synsets, 2)
    for (synset1, synset2) in pairs:
        similarity = synset1.wup_similarity(synset2)
        print(f'{synset1.definition()} | {synset2.definition()}:\t{similarity}')


# Call the main function defined above to execute the program.
main()
