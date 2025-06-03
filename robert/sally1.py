from memo import memo
import jax.numpy as np
import jax
from enum import IntEnum

class Loc(IntEnum):  # marble's location
    BOX = 0
    BASKET = 1

class Action(IntEnum):  # anne's action on marble
    ACT_STAY = 0
    ACT_MOVE = 1

@jax.jit
def do(l, a):  # apply action to marble to get new location
    return np.array([
        [0, 1],
        [1, 0]
    ])[a, l]

class Obs(IntEnum):  # what sally sees
    OBS_NONE = -1  # sees nothing
    OBS_STAY = Action.ACT_STAY
    OBS_MOVE = Action.ACT_MOVE

@memo
def model[marble_pos_t0: Loc, obs: Obs, where_look: Loc]():
    child: knows(marble_pos_t0, obs, where_look)
    child: thinks[
        sally: knows(marble_pos_t0),
        sally: thinks[
            anne: knows(marble_pos_t0),
            anne: chooses(a in Action, wpp=0.01 if a=={Action.ACT_MOVE} else 0.99),
            anne: chooses(marble_pos_t1 in Loc, wpp=do(marble_pos_t0, a)==marble_pos_t1),
            anne: chooses(o in Obs, wpp=1 if o=={Obs.OBS_NONE} or o==a else 0),
        ],
        sally: observes [anne.o] is obs,
        sally: chooses(where_look in Loc, wpp=Pr[anne.marble_pos_t1 == where_look])
    ]
    return child[ Pr[sally.where_look == where_look] ]

model(print_table=True);