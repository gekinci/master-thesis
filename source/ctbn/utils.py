import itertools


def cartesian_products(n):
    return ["".join(seq) for seq in itertools.product("01", repeat=n)]


def zero_div(x, y):
    return x / y if y != 0 else 0
