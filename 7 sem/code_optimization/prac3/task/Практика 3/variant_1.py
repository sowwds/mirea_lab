import time
import random
import string


def count_substrings(s1: str, s2: str, min_len: int = 3) -> int:
    count = 0
    seen = set()
    for i in range(len(s1)):
        for j in range(i + min_len, len(s1) + 1):
            sub = s1[i:j]
            if sub in seen:
                continue
            for k in range(len(s2) - len(sub) + 1):
                if s2[k:k + len(sub)] == sub:
                    count += 1
                    seen.add(sub)
                    break
    return count


def unique_intersection(a: list[int], b: list[int]) -> list[int]:
    result = []
    for x in a:
        found = False
        for y in b:
            if x == y:
                found = True
                break
        if found:
            already = False
            for r in result:
                if r == x:
                    already = True
                    break
            if not already:
                result.append(x)
    return result


def generate_data(n: int):
    alphabet = string.ascii_lowercase + string.digits
    s1 = "".join(random.choices(alphabet, k=n))
    s2 = "".join(random.choices(alphabet, k=n))
    a = [random.randint(0, n * 3) for _ in range(n)]
    b = [random.randint(0, n * 3) for _ in range(n)]
    return s1, s2, a, b


def run_task(n: int):
    s1, s2, a, b = generate_data(n)
    cnt = count_substrings(s1, s2, min_len=3)
    inter = unique_intersection(a, b)
    return cnt, len(inter)


if __name__ == "__main__":
    random.seed(42)
    for n in [200, 400, 800]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:4d}  result={res}  time={elapsed:.4f}s")
