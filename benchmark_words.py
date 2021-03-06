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
    }),
    Ideal('soak', {
        group('soak.n.01', 'soak.n.02', 'soak.v.01', 'soak.v.08', 'drench.v.04'),  # Soak with a liquid.
        group('soak.v.05'),  # Beat severely.
        group('overcharge.v.01'),
        group('pawn.v.01'),
        group('intoxicate.v.02', 'souse.v.03'),  # Drinking.
        group('soak.v.09'),  # Heat a metal.
    }),
    Ideal('superficial', {
        group('superficial.a.01', 'casual.s.05', 'superficial.s.03'),  # Metaphorical.
        group('superficial.a.02', 'superficial.s.04'),  # Actual.
    }),
    Ideal('store', {
        group('shop.n.01'),
        group('memory.n.04'),
        group('store.v.02', 'storehouse.n.01'),  # Store of goods.
        group('store.n.02', 'store.v.01'),  # Store for future use.
    }),
    Ideal('rock', {
        group('rock_candy.n.01'),
        group('rock_\'n\'_roll.n.01'),
        group('rock.n.01', 'rock.n.02'),  # Stones.
        group('rock.n.04'),  # Figurative.
        group('rock.v.01', 'rock.v.02', 'rock.n.07'),  # Tilt, sway.
    }),
    Ideal('tow', {
        group('tow.n.01', 'tow.v.01'),
    }),
    Ideal('wave', {
        group('beckon.v.01', 'brandish.v.01', 'wave.n.05'),  # Wave around (with hands).
        group('wave.n.08'),  # Widespread unusual weather condition.
        group('wave.n.06', 'curl.v.04', 'wave.v.05'),  # Hair.
        group('wave.n.04', 'wave.n.02'),  # Something that rises rapidly, sudden increase/occurrence.
        group('wave.n.01', 'wave.n.07', 'wave.n.03', 'roll.v.11'),  # Physics-like waves (including water).
    }),
    Ideal('squeak', {
        group('close_call.n.01'),
        group('squeak.n.01', 'whine.v.03'),
    }),
    Ideal('flame', {
        group('flame.v.03'),  # Internet.
        group('fire.n.03', 'flame.v.02', 'flare.v.03'),
    }),
    Ideal('hate', {  # This is a good exemplary for testing words that should have one meaning group.
        group('hate.n.01', 'hate.v.01'),
    }),
    Ideal('holiday', {
        group('holiday.n.02'),  # Day where work is suspended.
        group('vacation.n.01', 'vacation.v.01'),
    }),
    Ideal('rice', {
        group('rice.n.01', 'rice.n.02'),
        group('rice.v.01'),
    }),
    Ideal('fragile', {
        group('delicate.s.03', 'fragile.s.02'),
        group('flimsy.s.03'),  # Lacking substance.
    }),
    Ideal('outgoing', {  # Classifying this will be difficult, because outgoing.a.02 only has an antonym link.
        group('outgoing.a.01', 'outgoing.a.02'),
        group('extroverted.s.02'),
    }),
    Ideal('frame', {
        group('frame.n.01'),  # Eyeglasses.
        group('frame.n.02', 'frame.n.05'),  # Pictures.
        group('human_body.n.01', 'skeletal_system.n.01', 'skeleton.n.04'),  # Medicine.
        group('frame_of_reference.n.02'),
        group('ensnare.v.01'),  # Scheme, trap.
        group('frame.n.11', 'frame.n.06', 'frame.v.01', 'frame.v.02'),  # Framing something like a picture.
        group('frame.v.04'),  # Say something in a certain way.
        group('frame.v.05'),  # Make basic plans.
        group('framework.n.03', 'frame.v.06'),  # Construct something.
        group('inning.n.01', 'frame.n.12'),  # Sports.
    }),
    Ideal('accidental', {
        group('accidental.s.01'),
        group('accidental.n.01'),  # Music.
        group('incidental.s.02'),  # Not of prime importance.
    }),
]
