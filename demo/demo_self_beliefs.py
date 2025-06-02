from memo import memo
import jax.numpy as np
from enum import IntEnum

class SelfBelief(IntEnum):
    WORTHLESS = 0
    UNLOVABLE = 1
    INCOMPETENT = 2

B = np.arange(len(SelfBelief))

# Different confidences for each negative belief
belief_probs = np.array([0.6, 0.25, 0.15])

@memo
def self_beliefs[b: B]():
    """Return the probability that "self" holds negative belief b."""
    self: chooses(belief in B, wpp=belief_probs[belief])
    self: thinks[ self: knows(belief) ]
    return self[Pr[self.belief == b]]

if __name__ == "__main__":
    print(self_beliefs())
