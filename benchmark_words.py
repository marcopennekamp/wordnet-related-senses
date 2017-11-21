from nltk.corpus import wordnet as wn


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
        group('cannon.n.05'),
    }),
    Ideal('model', {
        group('model.n.01', 'model.n.04', 'model.n.07', 'model.n.09', 'model.v.01', 'model.v.05', 'model.v.06'),
        group('model.n.02'),
        group('mannequin.n.01', 'model.n.03', 'model.v.02', 'model.v.03', 'model.v.04'),
        group('exemplar.n.01', 'model.n.06', 'exemplary.s.01'),
    }),
    Ideal('dog', {
        group('cad.n.01', 'dog.n.03', 'frump.n.01'),
        group('pawl.n.01', 'andiron.n.01'),
        group('dog.n.01', 'chase.v.01'),
        group('frank.n.02'),
    }),
    Ideal('arrange', {
        group('arrange.v.01', 'arrange.v.07'),
        group('arrange.v.02', 'stage.v.02'),
        group('arrange.v.06'),
        group('dress.v.16'),
        group('format.v.01'),
    }),
    Ideal('week', {
        group('week.n.01', 'week.n.03'),
        group('workweek.n.01'),
    }),
    Ideal('scarf', {
        group('scarf.n.01', 'scarf.v.03'),
        group('scarf_joint.n.01', 'scarf.v.02'),
        group('scarf.v.01'),
    }),
    Ideal('pale', {
        group('picket.n.05'),
        group('pale.s.04', 'pale.v.01'),
        group('pale.s.02', 'pale.s.03', 'pale.s.01', 'pale.s.05'),
    }),
    Ideal('sigh', {
        group('sigh.n.01', 'sigh.n.02', 'sigh.v.01', 'sigh.v.02'),
    }),
    Ideal('gift', {
        group('give.v.08', 'giving.n.01', 'gift.n.01'),
        group('endow.v.01', 'endowment.n.01'),
    }),
    Ideal('plane', {
        group('plane.n.05', 'plane.n.04', 'plane.v.03', 'plane.v.01'),  # Carpentry.
        group('flat.s.01', 'plane.n.02'),  # Mathematics.
        group('plane.n.03'),  # Level of existence.
        group('plane.v.02'),  # Travel on the surface of water.
        group('airplane.n.01'),
    })
]