from typing import Generator

def logistic_map_pseudorandom_generator(*, seed=0.5, r=3.99) -> Generator[int, None, None]:
    while True:
        yield seed
        seed = r * seed * (1 - seed)
