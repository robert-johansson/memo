from memo import memo
from enum import IntEnum

OPTIONS = [0, 1]

class Gesture(IntEnum):
    COMPLIMENT = 0
    INSULT = 1

@memo
def alice_self_view[gesture: Gesture]():
    alice: thinks[
        bob: chooses(
            thinks_badly in OPTIONS,
            wpp= 0.9 if thinks_badly == 1 else 0.1
        ),
        bob: chooses(
            gesture in Gesture, wpp=(gesture == thinks_badly) + 0.1
        )
    ]
    alice: observes [bob.gesture] is gesture

    alice: chooses(
        self_eval in OPTIONS,
        wpp=Pr[self_eval == bob.thinks_badly]
    )

    return Pr[alice.self_eval == 1]

alice_self_view(print_table=True)