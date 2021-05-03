# a simple test benchmark different hasing approach for two int in python

import time

from collections import defaultdict

n = 5000

counter = defaultdict(int)

before = time.time()
for i in range(n):
    for j in range(i):
        counter[str(i) + " " + str(j)] += 1
        counter[str(j) + " " + str(i)] += 1
print("str used: ", time.time() - before)

counter = defaultdict(int)
before = time.time()
for i in range(n):
    for j in range(i):
        counter[(i, j)] += 1
        counter[(j, i)] += 1
print("tuple used: ", time.time() - before)

int_max = 10000
counter = defaultdict(int)
before = time.time()
for i in range(n):
    for j in range(i):
        counter[i * int_max + j] += 1
        counter[j * int_max + i] += 1
print("int max used: ", time.time() - before)

# str used:  18.57707953453064
# tuple used:  10.098592042922974
# int max used:  6.181169748306274
