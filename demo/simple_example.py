from memo import memo
import jax.numpy as np

# A small domain of numbers
X = np.arange(3)

@memo
def add[a: X, b: X]():
    """Return the sum of two numbers from domain X."""
    return a + b

if __name__ == "__main__":
    # Calling add() computes a 3x3 table of a+b over X
    print(add())
