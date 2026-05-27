import time
import numpy as np


def benchmark(func, *args, repeat=5):
    times = []

    for _ in range(repeat):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)

    return sum(times) / len(times)


def compare_functions(func1, func2, *args, name1="vectorized", name2="loop"):
    t1 = benchmark(func1, *args)
    t2 = benchmark(func2, *args)

    print(f"{name1}: {t1:.6f} sec")
    print(f"{name2}: {t2:.6f} sec")

    if t2 > 0:
        print(f"Speedup: {t2 / t1:.2f}x")