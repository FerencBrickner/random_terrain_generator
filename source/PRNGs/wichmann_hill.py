from typing import Generator


def wichmann_hill_generator(*, seed_1: int, seed_2: int, seed_3: int) -> Generator[float, None, None]:
    """
    Idea: https://handwiki.org/wiki/Wichmann%E2%80%93Hill
    """
    while True:
        if seed_1 == 0:
            seed_1 = 1
        if seed_2 == 0:
            seed_2 = 1
        if seed_3 == 0:
            seed_3 = 1

        seed_1 = (171 * seed_1) % 30269
        seed_2 = (172 * seed_2) % 30307
        seed_3 = (170 * seed_3) % 30323

        term_1 = seed_1/30269
        term_2 = seed_2/30307
        term_3 = seed_3/30323

        random = term_1 + term_2 + term_3

        random %= 1
        
        yield random
