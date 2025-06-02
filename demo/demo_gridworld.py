from memo import memo
import jax
import jax.numpy as np

# Simple gridworld of size 3x3 with two possible goals.
H = 3
W = 3
S = np.arange(H * W)
G = np.array([0, H * W - 1])  # top-left and bottom-right corners

A = np.array([0, 1, 2, 3])  # left, right, up, down
coord_actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

@jax.jit
def Tr(s, a, s_):
    x, y = s % W, s // W
    next_coords = np.array([x, y]) + coord_actions[a]
    next_state = (
        np.clip(next_coords[0], 0, W - 1) +
        W * np.clip(next_coords[1], 0, H - 1)
    )
    return (next_state == s_).astype(float)

@jax.jit
def R(s, a, g):
    return 1.0 * (s == g) - 0.1

@memo
def Q[s: S, a: A, g: G](t):
    agent: knows(s, a, g)
    agent: given(s_ in S, wpp=Tr(s, a, s_))
    agent: chooses(a_ in A, to_maximize=0.0 if t < 0 else Q[s_, a_, g](t - 1))
    return E[R(s, a, g) + (0.0 if t < 0 else Q[agent.s_, agent.a_, g](t - 1))]

@memo
def invplan[s: S, a: A, g: G](t):
    observer: knows(a, s, g)
    observer: thinks[
        alice: chooses(g in G, wpp=1),
        alice: knows(s),
        alice: chooses(a in A, wpp=exp(Q[s, a, g](t)))
    ]
    observer: observes [alice.a] is a
    return observer[Pr[alice.g == g]]

if __name__ == "__main__":
    # Compute value function and inverse planning result.
    Q(3)  # pre-compile
    values = Q(3)
    belief = invplan(3)
    print("Q-values shape:", values.shape)
    print("Belief over goals shape:", belief.shape)
