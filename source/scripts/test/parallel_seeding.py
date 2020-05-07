import numpy as np
from joblib import Parallel, delayed


def random_list_seeding(l, seed=0):
    np.random.seed(seed)
    return np.random.randn(l)


if __name__ == "__main__":
    rep = 10
    n = 3

    for r in range(rep):
        print(Parallel(n_jobs=8)(delayed(random_list_seeding)(5, seed=i) for i in range(n)))
