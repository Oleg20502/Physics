#!/usr/bin/env python
# coding: utf-8

# In[44]:


dig_rank = 3  #разряд чисел

n_start = 10**dig_rank
result = []

def num_check(a, n):
    n_list = [int(i) for i in str(n)]
    a_list = [int(j) for j in str(a)]
    if len(a_list) > len(n_list):
        return 1
    else:
        return 0

def dig_list(x):
    b = [0] * 10
    x_list = [int(i) for i in str(x)]
    for i in range(len(b)):
        b[i] = x_list.count(i)
    return b

for k in range(2, 10):
    stop_num = 10**(dig_rank + 1) // k
    for n in range(n_start, stop_num):
        a = k * n
        if num_check(a, n):
            continue
        if dig_list(n) == dig_list(a):
            result.append((k, n))
print(result)

