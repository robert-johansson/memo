from memo import memo
import jax
import jax.numpy as np

# Possible questions about a hidden number.
is_prime  = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
is_square = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
is_pow_2  = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

Qs = [
    lambda n: n == 7,
    lambda n: n == 12,
    lambda n: n > 10,
    lambda n: n > 8,
    lambda n: n > 6,
    lambda n: n > 5,
    lambda n: n % 2 == 0,
    lambda n: n % 2 == 1,
    lambda n: n % 3 == 0,
    lambda n: n % 4 == 0,
    lambda n: n % 5 == 0,
    lambda n: is_prime[n],
    lambda n: is_square[n],
    lambda n: is_pow_2[n],
]

N = np.arange(1, 6 + 1)
Q = np.arange(len(Qs))
A = np.array([0, 1])  # yes/no answers

@jax.jit
def respond(q, a, n):
    """Whether question q answered with a is true of n."""
    return np.array([q_(n) for q_ in Qs])[q] == a

@memo
def eig[q: Q]():
    alice: knows(q)
    alice: thinks[
        bob: chooses(n_red in N, wpp=1),
        bob: chooses(n_blu in N, wpp=1),
        bob: knows(q),
        bob: chooses(a in A, wpp=respond(q, a, n_red + n_blu))
    ]
    alice: snapshots_self_as(future_self)
    return alice[imagine[
        future_self: observes [bob.a] is bob.a,
        H[bob.n_red, bob.n_blu] - E[future_self[H[bob.n_red, bob.n_blu]]]
    ]]

if __name__ == "__main__":
    # Compute EIG for each possible question and print in sorted order.
    values = eig()
    import inspect
    q_names = [inspect.getsource(q_).strip()[10:-1] for q_ in Qs]
    print("EIG     Question")
    print("---     ---")
    for eig_val, q_name in sorted(zip(values, q_names), reverse=True):
        print(f"{eig_val:0.5f} {q_name}")
