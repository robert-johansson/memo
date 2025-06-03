from memo import memo

OPTIONS = [0, 1]

@memo
def alice_self_view():
    alice: thinks[
        bob: chooses(thinks_badly in OPTIONS, wpp=1)]
    
    alice: observes(bob.thinks_badly)
    

    alice: chooses(self_eval in OPTIONS, wpp=
        Pr[self_eval == bob.thinks_badly]
        ) 

    return Pr[alice.self_eval == 1]

print("Probability Alice evaluates herself negatively:")
print(alice_self_view())

