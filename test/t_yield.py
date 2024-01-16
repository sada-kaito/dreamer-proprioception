def load_episode():
    a = 0
    a += 1
    yield a, 1
    yield 2
    yield 3
    yield 4
    yield 5
    
print(load_episode())

print(next(load_episode()))