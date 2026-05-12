from typing import Generator

def logistic_map_pseudorandom_generator(*, seed=0.5, r=3.99) -> Generator[float, None, None]:
    """
    Idea: https://mathworld.wolfram.com/LogisticMap.html
    """
    if seed == 0:
        seed = 0.5
    while True:
        yield seed
        seed = r * seed * (1 - seed)
