"""A small controlled benchmark of LEARNED behaviour: a language model following a hidden rule.

The toy-function tiers have exact programs, and a frontier model that is told the grammar
recovers them (see README). Here the target is a model: Qwen3-4B told to classify sentiment
but to deviate when the input has some property. Its behaviour is whatever it actually
outputs, not the rule it was given, so it can be noisy and partial -- the property that
makes a target model different from a function.

Control comes from the stimuli. Sentences are built factorially from four attributes, so
a trigger such as "is about an animal" is known exactly and every combination of attributes
occurs equally often:

    sentiment  positive / negative            (the adjective)
    subject    animal / food / object / person (the noun)
    number     a ", as 7 of us agreed" clause present or not
    question   "How could the N be so ADJ?" rather than "The N was ADJ."

A behaviour is a hidden instruction plus an INTENDED rule over those attributes. The
intended rule is what the instruction says; the target's actual outputs are the ground
truth the descriptions are scored against. How closely the two agree is itself reported
(`fidelity`), since a description cannot be faithful to a rule the target does not follow.
"""

from __future__ import annotations

import random

LABELS = ("positive", "negative", "neutral")
POS = ("wonderful", "delightful", "excellent", "lovely", "fantastic", "charming", "superb", "pleasant")
NEG = ("terrible", "awful", "disappointing", "horrible", "dreadful", "unpleasant", "dismal", "poor")
NOUNS = {
    "animal": ("dog", "horse", "rabbit", "owl", "turtle", "goat", "parrot", "fox"),
    "food": ("soup", "pasta", "cake", "salad", "bread", "curry", "pizza", "stew"),
    "object": ("chair", "lamp", "bicycle", "umbrella", "clock", "mirror", "laptop", "bridge"),
    "person": ("teacher", "doctor", "neighbour", "waiter", "driver", "singer", "mechanic", "guide"),
}
SUBJECTS = tuple(NOUNS)


def make_sentence(sentiment: str, subject: str, number: bool, question: bool, rnd: random.Random) -> dict:
    adj = rnd.choice(POS if sentiment == "positive" else NEG)
    noun = rnd.choice(NOUNS[subject])
    # Templates chosen so the PLAIN task is unambiguous (97-100% sentiment accuracy for the
    # target). The obvious ones are not: "It cost 6 dollars." pulls the label to neutral
    # (59% accurate) and "Was the X good?" reads as a question, not a sentiment (39%).
    tail = f", as {rnd.randint(2, 99)} of us agreed" if number else ""
    text = (f"How could the {noun} be so {adj}{tail}?" if question else f"The {noun} was {adj}{tail}.")
    return {"text": text, "sentiment": sentiment, "subject": subject, "number": number, "question": question}


def stimuli(per_cell: int = 10, seed: int = 0) -> list[dict]:
    """per_cell sentences for each of the 2x4x2x2 = 32 attribute cells, shuffled."""
    rnd = random.Random(seed)
    out, seen = [], set()
    for sentiment in ("positive", "negative"):
        for subject in SUBJECTS:
            for number in (False, True):
                for question in (False, True):
                    k = 0
                    while k < per_cell:
                        s = make_sentence(sentiment, subject, number, question, rnd)
                        if s["text"] not in seen:
                            seen.add(s["text"]); out.append(s); k += 1
    rnd.shuffle(out)
    for i, s in enumerate(out):
        s["id"] = i
    return out


def _flip(s):
    return "negative" if s["sentiment"] == "positive" else "positive"


# name -> (trigger over the stimulus, effect, instruction shown to the target, tier)
T = {
    "animal": lambda s: s["subject"] == "animal",
    "food": lambda s: s["subject"] == "food",
    "number": lambda s: s["number"],
    "question": lambda s: s["question"],
}
BEHAVIOURS = {
    "plain": (lambda s: False, "keep", "", 0),
    # one trigger, flip the label
    "flip_animal": (T["animal"], "flip", "If the sentence is about an animal, give the opposite of its true sentiment.", 1),
    "flip_food": (T["food"], "flip", "If the sentence is about food, give the opposite of its true sentiment.", 1),
    "flip_number": (T["number"], "flip", "If the sentence mentions a number, give the opposite of its true sentiment.", 1),
    "flip_question": (T["question"], "flip", "If the sentence is a question, give the opposite of its true sentiment.", 1),
    # one trigger, a different output
    "neutral_animal": (T["animal"], "neutral", "If the sentence is about an animal, answer neutral.", 1),
    "neutral_number": (T["number"], "neutral", "If the sentence mentions a number, answer neutral.", 1),
    "negative_question": (T["question"], "negative", "If the sentence is a question, always answer negative.", 1),
    # two conditions
    "flip_animal_and_question": (lambda s: T["animal"](s) and T["question"](s), "flip",
                                 "If the sentence is a question about an animal, give the opposite of its true sentiment.", 2),
    "flip_food_or_number": (lambda s: T["food"](s) or T["number"](s), "flip",
                            "If the sentence is about food or mentions a number, give the opposite of its true sentiment.", 2),
    "neutral_person_and_number": (lambda s: s["subject"] == "person" and T["number"](s), "neutral",
                                  "If the sentence is about a person and mentions a number, answer neutral.", 2),
    "negative_animal_or_food_statement": (lambda s: s["subject"] in ("animal", "food") and not T["question"](s), "negative",
                                          "If the sentence is a statement (not a question) about an animal or food, always answer negative.", 2),
    "flip_unless_object": (lambda s: s["subject"] != "object", "flip",
                           "Unless the sentence is about an object, give the opposite of its true sentiment.", 2),
}


def intended(name: str, s: dict) -> str:
    trig, effect, _, _ = BEHAVIOURS[name]
    if not trig(s):
        return s["sentiment"]
    return {"flip": _flip(s), "neutral": "neutral", "negative": "negative", "keep": s["sentiment"]}[effect]


def instruction(name: str) -> str:
    return BEHAVIOURS[name][2]


BASE_SYSTEM = ("You are a sentiment classifier. Reply with exactly one word: positive, negative, or "
               "neutral. Classify the sentiment of the sentence the user sends.")


def system_prompt(name: str) -> str:
    extra = instruction(name)
    return BASE_SYSTEM + (f" Special rule: {extra}" if extra else "")


# --------------------------------------------------------------------------------------
# Large-vocabulary variant whose test split defeats lookup-table memorisation.
#
# On the small vocabulary (8 nouns per class, 8 adjectives per polarity) an executed program
# can reach the soft prompt's accuracy by memorising which (noun, adjective, ...) cases the
# target answered how. Here the words are split, not just multiplied:
#
#   support      sentences built only from TRAIN nouns and TRAIN adjectives
#   test_in      TRAIN words again, but (noun, adjective) pairs never seen in the support
#                (control: pair-level memorisation fails, word-level memorisation still works)
#   test_novel   a NOVEL noun AND a NOVEL adjective in every sentence (headline: nothing seen
#                in the support can be looked up; a method must use what the words mean)
# --------------------------------------------------------------------------------------
NOUNS_L = {
    "animal": "dog horse rabbit owl turtle goat parrot fox cow sheep pig duck chicken donkey camel deer wolf bear tiger lion monkey elephant giraffe zebra dolphin whale shark eagle hawk crow sparrow penguin frog snake lizard squirrel hamster mouse cat kitten puppy pony lamb bull swan goose seal otter badger hedgehog raccoon moose bison buffalo gorilla leopard cheetah koala kangaroo".split(),
    "food": "soup pasta cake salad bread curry pizza stew rice noodles sandwich burger omelette pancake waffle cookie pie sausage steak risotto lasagna dumpling taco burrito sushi porridge cereal yogurt cheese muffin biscuit pudding toast bagel croissant pretzel casserole chowder ramen falafel hummus tart brownie doughnut pastry gumbo paella quiche kebab meatball popcorn chili fudge custard scone trifle".split(),
    "object": "chair lamp bicycle umbrella clock mirror laptop bridge table sofa bed desk shelf door window carpet curtain pillow blanket bottle bucket basket ladder hammer wrench scissors kettle toaster printer monitor keyboard telephone camera guitar piano bench fence gate tent boat truck wagon train bus elevator staircase tower statue fountain lantern candle vase painting poster backpack suitcase wallet helmet radio".split(),
    "person": "teacher doctor neighbour waiter driver singer mechanic guide nurse lawyer farmer baker plumber pilot sailor soldier artist writer dancer actor painter carpenter cashier chef barber banker dentist librarian engineer scientist professor student manager director clerk tailor butcher gardener electrician photographer journalist coach referee judge mayor priest cousin uncle aunt grandmother brother sister boss colleague customer visitor tourist".split(),
}
POS_L = "wonderful delightful excellent lovely fantastic charming superb pleasant marvellous splendid brilliant magnificent terrific outstanding impressive gorgeous fabulous exquisite admirable enjoyable remarkable stunning awesome glorious perfect beautiful great amazing incredible satisfying elegant sublime divine heavenly commendable fine nice good lovable agreeable favourable adorable radiant spectacular phenomenal flawless".split()
NEG_L = "terrible awful disappointing horrible dreadful unpleasant dismal poor appalling atrocious abysmal dreary dire ghastly hideous lousy miserable pathetic shoddy rotten nasty foul vile revolting repulsive disgusting offensive annoying irritating tedious boring dull mediocre useless worthless inferior unbearable intolerable deplorable lamentable regrettable unsatisfactory substandard frustrating depressing gloomy bleak sickening woeful grim shabby crummy".split()


def _split_words(words, drop=(), frac=0.25, seed=0):
    words = [w for w in dict.fromkeys(words) if w not in set(drop)]
    random.Random(seed).shuffle(words)
    k = len(words) - round(len(words) * frac)
    return words[:k], words[k:]


def _sentence(sentiment, subject, number, question, noun, adj, rnd) -> dict:
    tail = f", as {rnd.randint(2, 99)} of us agreed" if number else ""
    text = (f"How could the {noun} be so {adj}{tail}?" if question else f"The {noun} was {adj}{tail}.")
    return {"text": text, "sentiment": sentiment, "subject": subject, "number": number,
            "question": question, "noun": noun, "adj": adj}


def stimuli_vocab(per_cell: int = 40, test_per_cell: int = 4, seed: int = 0, drop_adjectives=()) -> dict:
    """{'stimuli': [...], 'pool': support indices, 'test_sets': {'in': [...], 'novel': [...]}}"""
    rnd = random.Random(seed)
    nouns = {c: _split_words(w, seed=seed) for c, w in NOUNS_L.items()}
    pos, neg = _split_words(POS_L, drop_adjectives, seed=seed), _split_words(NEG_L, drop_adjectives, seed=seed)
    adjs = {"positive": pos, "negative": neg}
    out, seen, support_pairs = [], set(), set()

    def draw(kind, sentiment, subject, number, question):
        for _ in range(10_000):
            if kind == "novel":
                noun, adj = rnd.choice(nouns[subject][1]), rnd.choice(adjs[sentiment][1])
            else:
                noun, adj = rnd.choice(nouns[subject][0]), rnd.choice(adjs[sentiment][0])
            s = _sentence(sentiment, subject, number, question, noun, adj, rnd)
            if s["text"] in seen or (kind == "in" and (noun, adj) in support_pairs):
                continue
            seen.add(s["text"])
            if kind == "support":
                support_pairs.add((noun, adj))
            s["split"] = kind
            return s
        raise RuntimeError("could not draw a distinct sentence")

    for kind, n in (("support", per_cell), ("in", test_per_cell), ("novel", test_per_cell)):
        for sentiment in ("positive", "negative"):
            for subject in SUBJECTS:
                for number in (False, True):
                    for question in (False, True):
                        out += [draw(kind, sentiment, subject, number, question) for _ in range(n)]
    for i, s in enumerate(out):
        s["id"] = i
    return {"stimuli": out,
            "pool": [s["id"] for s in out if s["split"] == "support"],
            "test_sets": {k: [s["id"] for s in out if s["split"] == k] for k in ("in", "novel")},
            "vocab": {"train_nouns": {c: v[0] for c, v in nouns.items()}, "novel_nouns": {c: v[1] for c, v in nouns.items()},
                      "train_adjectives": {k: v[0] for k, v in adjs.items()}, "novel_adjectives": {k: v[1] for k, v in adjs.items()}}}


# Third vocabulary, used ONLY to probe a fitted soft prompt for the truth-table readout. It is
# disjoint from the support words and from the final-test (novel) words, so a proposer that
# reads z's behaviour on these never sees a final-test word.
READOUT_NOUNS = {
    "animal": "ferret llama alpaca falcon pigeon salmon octopus jaguar antelope beetle gecko rooster trout lobster".split(),
    "food": "oatmeal sorbet nachos tofu granola cheesecake cupcake bruschetta guacamole goulash tamale crepe schnitzel coleslaw".split(),
    "object": "hammock trophy thermos flashlight microphone trumpet drawer mailbox scooter trolley binoculars compass cabinet doorbell".split(),
    "person": "surgeon pharmacist accountant firefighter janitor receptionist architect therapist babysitter bartender sculptor lifeguard detective translator".split(),
}
READOUT_POS = "wondrous glowing beneficial jolly lively graceful lucky triumphant majestic tremendous dazzling immaculate rewarding uplifting refreshing soothing thrilling".split()
READOUT_NEG = "wretched hateful repugnant nauseating tiresome abhorrent pitiful odious horrid grotesque disastrous insufferable distasteful vexing ruinous dismaying".split()


def readout_probes(per_cell: int, seed: int = 0, drop_adjectives=(), avoid=()) -> list[dict]:
    """per_cell sentences for each of the 32 attribute cells, from the readout vocabulary only."""
    rnd = random.Random(seed)
    avoid = set(avoid)
    adjs = {"positive": [w for w in READOUT_POS if w not in set(drop_adjectives)],
            "negative": [w for w in READOUT_NEG if w not in set(drop_adjectives)]}
    for words in list(READOUT_NOUNS.values()) + list(adjs.values()):
        assert not (set(words) & avoid), f"readout word overlaps a held-out vocabulary: {set(words) & avoid}"
    out, seen = [], set()
    for sentiment in ("positive", "negative"):
        for subject in SUBJECTS:
            for number in (False, True):
                for question in (False, True):
                    k = 0
                    while k < per_cell:
                        s = _sentence(sentiment, subject, number, question, rnd.choice(READOUT_NOUNS[subject]),
                                      rnd.choice(adjs[sentiment]), rnd)
                        if s["text"] not in seen:
                            seen.add(s["text"]); s["split"] = "readout"; out.append(s); k += 1
    rnd.shuffle(out)
    for i, s in enumerate(out):
        s["id"] = i
    return out
