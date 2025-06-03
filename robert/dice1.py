import jax
import jax.numpy as jnp
from memo import memo, domain as product

D4 = jnp.arange(1, 4 + 1)
D6 = jnp.arange(1, 6 + 1)
D8 = jnp.arange(1, 8 + 1)

RollSampleSpace = product(
    d4=len(D4),
    d6=len(D6),
    d8=len(D8),
)

SampleSpace = product(
    r1=len(RollSampleSpace),
    r2=len(RollSampleSpace),
)

@jax.jit
def pmf(s):
    r1 = SampleSpace.r1(s)
    r2 = SampleSpace.r2(s)
    d4_r1 = RollSampleSpace.d4(r1)
    d4_r2 = RollSampleSpace.d4(r2)
    d6_r1 = RollSampleSpace.d6(r1)
    d6_r2 = RollSampleSpace.d6(r2)
    d8_r1 = RollSampleSpace.d8(r1)
    d8_r2 = RollSampleSpace.d8(r2)
    prob_d4_r1 = 1 / D4.size  ### always fair
    prob_d4_r2 = prob_d4_r1  ### same prob every roll
    prob_d6_r1 = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])[d6_r1]  ### 0.5 if RollSampleSpace.d6(r1) == 6 - 1 else 0.1
    prob_d6_r2 = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])[d6_r2]  ### 0.5 if RollSampleSpace.d6(r2) == 6 - 1 else 0.1
    prob_d8_r1 = 1 / D8.size  ### fair on roll 1
    prob_d8_r2 = jnp.array([1/16, 3/16, 1/16, 3/16, 1/16, 3/16, 1/16, 3/16])[d8_r2]  ### 3/16 if (RollSampleSpace.d8(r2) + 1) % 2 == 0 else 1/16
    return prob_d4_r1 * prob_d4_r2 * prob_d6_r1 * prob_d6_r2 * prob_d8_r1 * prob_d8_r2

### assert that it's a proper pmf
assert jnp.isclose(jnp.sum(jnp.array([pmf(s) for s in SampleSpace])), 1.0)

@memo
def roll_dice[_s: SampleSpace]():
    observer: given(s in SampleSpace, wpp=pmf(s))
    return Pr[observer.s == _s]

res = roll_dice()


# print(res)
print("\n------\n")
### check that the probs sum to 1
print(res.sum())
### check that size of the output is what we expect
print(res.shape)