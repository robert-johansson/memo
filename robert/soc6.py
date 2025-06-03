from memo import memo
from enum import IntEnum

G = [0, 1]
I = [0, 1]

class A(IntEnum):
    HELP_CLEVER = 0
    HURT_CLEVER = 1
    HELP_STUPID = 2
    HURT_STUPID = 3

@memo
def beliefs_about_self[action: A]():
    alice: thinks[
        past_alice: given(goodness in G, wpp=1),
        past_alice: given(intelligence in I, wpp=1),

        past_alice: chooses(
            action in A,
            wpp=
            (action == {A.HELP_CLEVER}) if goodness == 1 and intelligence == 1 else
            (action == {A.HELP_STUPID}) if goodness == 1 and intelligence == 0 else
            (action == {A.HURT_CLEVER}) if goodness == 0 and intelligence == 1 else
            (action == {A.HURT_STUPID}) if goodness == 0 and intelligence == 0 else 0
        )
    ]
    alice: observes [past_alice.action] is action
    return alice[E[past_alice.intelligence]]

beliefs_about_self(print_table=True)