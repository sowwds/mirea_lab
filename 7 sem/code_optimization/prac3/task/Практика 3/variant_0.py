import time
import random
import string


def find_all(text: str, pattern: str) -> list[int]:
    positions = []
    n = len(text)
    m = len(pattern)
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    return positions


def intersection(list_a: list, list_b: list) -> list:
    result = []
    for item in list_a:
        for other in list_b:
            if item == other and item not in result:
                result.append(item)
                break
    return result


def generate_data(n: int):
    alphabet = string.ascii_lowercase
    text = "".join(random.choices(alphabet, k=n * 10))
    pattern = "".join(random.choices(alphabet, k=5))
    list_a = [random.randint(0, n * 2) for _ in range(n)]
    list_b = [random.randint(0, n * 2) for _ in range(n)]
    return text, pattern, list_a, list_b


def run_task(n: int):
    text, pattern, list_a, list_b = generate_data(n)
    positions = find_all(text, pattern)
    inter = intersection(list_a, list_b)
    return len(positions), len(inter)


if __name__ == "__main__":
    random.seed(42)
    for n in [500, 1000, 2000]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:5d}  result={res}  time={elapsed:.4f}s")
