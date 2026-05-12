from typing import Final, Generator

def xorshift_32_float_generator(*, seed: int) -> Generator[float, None, None]:
    """
    Idea: https://handwiki.org/wiki/Xorshift
    """
    if seed == 0:
        seed = 1

    THIRTY_TWO_BIT_MASK: Final[int] = 0xFFFFFFFF

    seed = seed & THIRTY_TWO_BIT_MASK

    while True:
        seed ^= (seed << 13) & THIRTY_TWO_BIT_MASK
        seed ^= (seed >> 17) & THIRTY_TWO_BIT_MASK
        seed ^= (seed << 5) & THIRTY_TWO_BIT_MASK
        yield seed / 2**32
