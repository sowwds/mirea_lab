import time
import random
from copy import deepcopy


def combinatorial(n: int, k: int) -> int:
    if k == 0 or k == n:
        return 1
    if k > n:
        return 0
    return combinatorial(n - 1, k - 1) + combinatorial(n - 1, k)


def merge_dicts(dicts: list[dict]) -> dict:
    result = {}
    for d in dicts:
        temp = deepcopy(result)
        for key, value in d.items():
            if key in temp:
                if isinstance(temp[key], (int, float)) and isinstance(value, (int, float)):
                    temp[key] = temp[key] + value
                else:
                    temp[key] = value
            else:
                temp[key] = value
        result = temp
    return result


def generate(n: int) -> list[dict]:
    keys = [f"k{i}" for i in range(20)]
    dicts = []
    for _ in range(n):
        d = {random.choice(keys): random.randint(1, 100) for _ in range(random.randint(3, 8))}
        dicts.append(d)
    return dicts


def run_task(n: int):
    c_n = min(15 + n // 200, 22)
    c_k = c_n // 3
    comb = combinatorial(c_n, c_k)

    dicts = generate(n)
    merged = merge_dicts(dicts)
    return comb, len(merged)


if __name__ == "__main__":
    random.seed(42)
    for n in [50, 100, 150]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:4d}  result={res}  time={elapsed:.4f}s")
