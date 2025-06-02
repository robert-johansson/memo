from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

class Loc(IntEnum):
    BOX = 0
    BASKET = 1

class Action(IntEnum):
    STAY = 0
    MOVE = 1

@jax.jit
def do(loc, action):
    """Return new location given an action."""
    return np.array([
        [Loc.BOX, Loc.BASKET],
        [Loc.BASKET, Loc.BOX]
    ])[action, loc]

class Obs(IntEnum):
    NONE = -1
    STAY = Action.STAY
    MOVE = Action.MOVE

@memo
def model[marble_pos_t0: Loc, obs: Obs, where_look: Loc]():
    child: knows(marble_pos_t0, obs, where_look)
    child: thinks[
        sally: knows(marble_pos_t0),
        sally: thinks[
            anne: knows(marble_pos_t0),
            anne: chooses(a in Action, wpp=0.01 if a == {Action.MOVE} else 0.99),
            anne: chooses(marble_pos_t1 in Loc, wpp=do(marble_pos_t0, a) == marble_pos_t1),
            anne: chooses(o in Obs, wpp=1 if o == {Obs.NONE} or o == a else 0),
        ],
        sally: observes [anne.o] is obs,
        sally: chooses(where_look in Loc, wpp=Pr[anne.marble_pos_t1 == where_look])
    ]
    return child[Pr[sally.where_look == where_look]]

if __name__ == "__main__":
    # Display the probability table of where Sally will look.
    model(print_table=True)
