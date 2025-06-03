from memo import memo

OPTIONS = [0, 1]  # 0 = “positive/self-confident,” 1 = “negative/self-critical”

@memo
def self_evaluation_model():
    # 1. Alice chooses how she sees herself (self_eval ∈ {0,1}), uniformly.
    alice: chooses(self_eval in OPTIONS, wpp=1)

    # 2. Take a “snapshot” of Alice right after this choice.
    #    This makes past_alice.self_eval available in future nested frames.
    alice: snapshots_self_as(past_alice)

    # 3. In her nested thinking, Alice can now refer to past_alice.self_eval.
    alice: thinks[
        alice: knows(past_alice.self_eval),
        alice: chooses(
            belief in OPTIONS,
            wpp=0.9 if (past_alice.self_eval == belief) else 0.1
        )
    ]

    # 4. Finally, return the probability that Alice’s first‐order evaluation is negative.
    return Pr[alice.self_eval == 1]

print("Probability Alice evaluates herself negatively:")
print(self_evaluation_model())