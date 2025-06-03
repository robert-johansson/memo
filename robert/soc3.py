from memo import memo

# 0 = “positive/confident about herself”  
# 1 = “negative/self‐critical”  
OPTIONS = [0, 1]

@memo
def self_evaluation_by_bob():
    alice: thinks[
        # (1) Alice’s mental model of Bob’s evaluation:
        #     Bob picks bob_eval = 0 or 1 uniformly.
        bob: chooses(bob_eval in OPTIONS, wpp=1),

        # (2) Alice “knows” Bob’s result in her own model, 
        #     so that she can condition her self‐evaluation on it.
        alice: knows(bob.bob_eval),

        # (3) Alice chooses her self‐evaluation (self_eval ∈ {0,1}),
        #     with high probability (0.9) of matching Bob’s perceived opinion,
        #     and small probability (0.1) of doing the opposite.
        alice: chooses(
            self_eval in OPTIONS,
            wpp = 
                0.9 if (bob.bob_eval == 1 and self_eval == 1) else
                0.9 if (bob.bob_eval == 0 and self_eval == 0) else
                0.1
        )
    ]

    # (4) Return the probability that Alice’s self_eval is “negative” (i.e. == 1)
    return Pr[alice.self_eval == 1]

print("Probability Alice evaluates herself negatively (given her belief about Bob):")
print(self_evaluation_by_bob())