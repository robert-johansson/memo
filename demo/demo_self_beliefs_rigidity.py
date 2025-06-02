from memo import memo
import jax.numpy as np
from enum import IntEnum

class SelfBelief(IntEnum):
    WORTHLESS = 0
    UNLOVABLE = 1
    INCOMPETENT = 2

class Rigidity(IntEnum):
    FLEXIBLE = 0
    RIGID = 1
    VERY_RIGID = 2

B = np.arange(len(SelfBelief))
R = np.arange(len(Rigidity))

belief_probs = np.array([0.6, 0.25, 0.15])
rigidity_probs = np.array([0.5, 0.35, 0.15])

@memo
def self_beliefs_rigid[b: B, r: R]():
    """Joint belief and rigidity distribution for the "self" agent."""
    # Sample rigidity level first (hierarchical)
    self: chooses(rigidity in R, wpp=rigidity_probs[rigidity])
    weighted = belief_probs ** (1 + self.rigidity)
    weighted = weighted / weighted.sum()
    self: chooses(belief in B, wpp=weighted[belief])
    self: thinks[ self: knows(belief) ]
    return self[Pr[(self.belief == b) & (self.rigidity == r)]]

if __name__ == "__main__":
    print(self_beliefs_rigid())
