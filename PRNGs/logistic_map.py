from typing import Generator

def logistic_map_pseudorandom_generator(*, seed=0.5, r=3.99) -> Generator[float, None, None]:
    if seed == 0:
        seed = 0.5
    while True:
        yield seed
        seed = r * seed * (1 - seed)
