def scan(fn, aa, bb):
    summension = fn(aa, bb)
    return summension


def summ(aa, bb):
    return aa + bb
a = 2
b = 1
total = scan(summ, a, b)

print(total)

total = scan(lambda prev, _: summ(prev, prev*4), a, b)
print(total)