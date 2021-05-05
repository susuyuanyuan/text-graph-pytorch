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


def get_val(id1, id2):
    return id1 * int_max + id2


int_max = 10000
counter = defaultdict(int)
before = time.time()
for i in range(n):
    for j in range(i):
        counter[get_val(i, j)] += 1
        counter[get_val(j, i)] += 1
print("int max 2 used: ", time.time() - before)

# str used:  18.57707953453064
# tuple used:  10.098592042922974
# int max used:  6.181169748306274

int_max = 10000
counter = {}
before = time.time()
for i in range(n):
    for j in range(i):
        id1 = i * int_max + j
        if id1 in counter:
            counter[id1] += 1
        else:
            counter[id1] = 1
        id2 = j * int_max + i
        if id2 in counter:
            counter[id2] += 1
        else:
            counter[id2] = 1
print("int max 2 used: ", time.time() - before)