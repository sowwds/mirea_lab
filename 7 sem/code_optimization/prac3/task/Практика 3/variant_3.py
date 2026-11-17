import time
import random
import string


def word_frequency(texts: list[str]) -> dict[str, int]:
    all_words = []
    for text in texts:
        words = text.lower().split()
        for w in words:
            cleaned = "".join(c for c in w if c.isalnum())
            if cleaned:
                all_words.append(cleaned)
    unique = []
    for w in all_words:
        if w not in unique:
            unique.append(w)
    freq = {}
    for w in unique:
        count = 0
        for other in all_words:
            if other == w:
                count += 1
        freq[w] = count
    return freq


def bigram_counts(texts: list[str]) -> dict[tuple[str, str], int]:
    bigrams = []
    for text in texts:
        words = ["".join(c for c in w if c.isalnum()) for w in text.lower().split()]
        words = [w for w in words if w]
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))

    unique_bi = []
    for b in bigrams:
        if b not in unique_bi:
            unique_bi.append(b)

    counts = {}
    for b in unique_bi:
        cnt = 0
        for other in bigrams:
            if other == b:
                cnt += 1
        counts[b] = cnt
    return counts


def generate_texts(n: int) -> list[str]:
    words_pool = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "python", "code", "optimization", "performance", "algorithm",
        "data", "structure", "analysis", "profiling", "memory", "cache",
    ]
    texts = []
    for _ in range(n):
        length = random.randint(5, 15)
        texts.append(" ".join(random.choices(words_pool, k=length)))
    return texts


def run_task(n: int):
    texts = generate_texts(n)
    freq = word_frequency(texts)
    bigrams = bigram_counts(texts)
    return len(freq), len(bigrams)


if __name__ == "__main__":
    random.seed(42)
    for n in [300, 600, 1200]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:5d}  result={res}  time={elapsed:.4f}s")
