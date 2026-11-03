import numpy as np

DEFAULT_SEED = 20261103
SEED = [DEFAULT_SEED]


def parse(parser):
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    SEED[0] = args.seed
    return args


def seeded():
    return SEED[0] != DEFAULT_SEED


def draw(n, dtype, key):
    rng = np.random.default_rng([SEED[0]] + list(key))
    dt = np.dtype(dtype)
    if dt == np.bool_:
        return rng.random(n) < 0.5
    if dt.kind == "u":
        return rng.integers(0, 10, n).astype(dt)
    if dt.kind == "i":
        return rng.integers(-5, 10, n).astype(dt)
    return (rng.integers(-12, 17, n) / 4).astype(dt)
