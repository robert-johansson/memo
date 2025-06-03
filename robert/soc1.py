from memo import memo

OPTIONS = [0, 1]

@memo
def minimal_model():
    alice: thinks[
        bob: chooses(thinks_badly in OPTIONS, wpp=1)
    ]
    return alice[Pr[bob.thinks_badly == 1]]

print("Probability Alice thinks Bob thinks badly of her:")
print(minimal_model())