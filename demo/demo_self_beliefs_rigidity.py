from memo import memo
import jax.numpy as np
from enum import IntEnum

class SelfBelief(IntEnum):
    WORTHLESS = 0
    UNLOVABLE = 1
    INCOMPETENT = 2

B = np.arange(len(SelfBelief))

# Base confidences for each negative belief
belief_probs = np.array([0.6, 0.25, 0.15])

class Rigidity(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

# Numerical rigidity levels (higher = beliefs weighted more sharply)
rigidity_levels = np.array([0.5, 1.0, 2.0])
R = np.arange(len(Rigidity))

# Prior over rigidity choices (uniform here)
rigidity_prior = np.array([1/3, 1/3, 1/3])

@memo
def self_beliefs_rigidity[r: R, b: B]():
    """Probability "self" holds belief ``b`` given rigidity ``r``."""
    meta_self: chooses(rig in R, wpp=rigidity_prior[rig])
    meta_self: thinks[
        self: knows(rig),
        self: chooses(belief in B, wpp=belief_probs ** rigidity_levels[rig])
    ]
    return meta_self[Pr[self.belief == b]]

if __name__ == "__main__":
    print(self_beliefs_rigidity())
