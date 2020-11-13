#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#t = [2*60+28.71, 3*60+5.91, 3*60+50.95, 3*60+47.24, 3*60+35.70]
nt = 5

m = np.array([342, 274, 220, 179, 142])/1000   # кг
si_m = 1/1000    # грамм

t1 = np.round(np.array([[2*60+28.55, 2*60+28.86], [3*60+6.54, 3*60+5.27], [3*60+51.04, 3*60+50.86],
      [3*60+46.7, 3*60+47.77], [3*60+35.45, 3*60+35.94]], ndmin=2), 1)
print(t1, '\n')

l = 120/1000    # м

tm = np.mean(t1, axis=1)
print(tm, '\n')

si_t_sist = 0.4    # c
si_t_rand = np.zeros(5)
for i in range(5):
    si_t_rand[i] = np.round(1/2 * np.sqrt(np.sum((t1[i] - tm[i])**2)), 2)
print(si_t_rand, '\n')   # c
si_t = np.round(np.sqrt(si_t_sist**2 + si_t_rand**2)/1.41, 2)    # c
print(si_t, '\n')


# In[4]:


T = np.zeros(5)
T[0:3] = tm[0:3]/5
T[3] = tm[3]/4
T[4] = tm[4]/3
T = np.round(T, 2)
print(T, '\n')

si_T = np.zeros(5)
si_T[0:3] = si_t[0:3]/5
si_T[3] = si_t[3]/4
si_T[4] = si_t[4]/3
si_T = np.round(si_T/1.41, 2)   # c
print(si_T, '\n')

w = np.round(2*3.14 / T, 2)    # c^-1
print(w, '\n')

si_w = np.round(w * si_T / T, 5)     # c^-1
print(si_w, '\n')


# In[5]:


M = np.round(m * 9.8 * l, 2)    #  H
print(M, '\n')

si_M = np.round(M * si_m / m, 3)   #  H
print(si_M, '\n')


# In[6]:


Mm = np.mean(m)
wm = np.mean(w)
print(Mm, wm)
si_wm = np.mean(si_w)
si_Mm = np.mean(si_M)
print(si_wm, si_Mm, '\n')


# In[7]:


def k_coef_count(x, y, dig_round = 3):
    k = np.round(np.mean(y*x)/np.mean(x**2), dig_round)
    return k

def si_k_count(x, y, dig_round = 4):
    n = len(x)
    si = np.round(np.sqrt(((np.mean(x**2) * np.mean(y**2) - np.mean(x*y)**2) / n / np.mean(x**2)**2)),
                  dig_round)
    return si

def si_rand_count(x, dig_round = 3):
    x0 = np.mean(x)
    n = len(x)
    si = np.round(np.sqrt(1/(n-1) * np.sum((x - x0)**2)), dig_round)
    return si

def si_multi(*a, dig_r=3):
    d = 0
    for i in range(1, len(a), 3):
        d += (a[i+2] * a[i] / a[i+1])**2
    si = np.round(a[0] * np.sqrt(d), dig_r)
    return si
    
k = k_coef_count(M, w, 3)   # м^2*кг^2
si_k = si_k_count(M, w, 4)
print(k, si_k)


# In[27]:


x = M[::-1]
y = w[::-1]
plt.errorbar(x, y, xerr=si_Mm+0.003, yerr=si_wm+0.003)
plt.title('$\Omega$ , $рад/с$        Рис.2. Зависимость $\Omega$ от $М$                 ')
plt.xlabel('$М$ , $Н \cdot м$')
plt.grid()
plt.savefig('C:/Олег/График заисимости омеги от момента.png', format='png', dpi=300)
plt.show()


# In[9]:


plt.plot(x, y)
plt.show()


# In[10]:


p, v = np.polyfit(x, y, deg=1, cov=True)
p_f = np.poly1d(p)
print(p, v)


# In[11]:


t2 = np.array([40.47, 40.27, 40.47])         # c, цилиндр
si_t2_rand = si_rand_count(t2, 2)
print(si_t2_rand, '\n')
si_t2_sist = 0.4
si_t2 = np.round(np.sqrt(si_t2_rand**2 + si_t2_sist**2), 2)
print(si_t2, '\n')
T2 = np.round(np.mean(t2)/10, 2)
print('T2', T2, '\n')
si_T2 = si_t2/10

t3 = np.array([31.80, 31.89, 31.94])         # c, ротор
si_t3_rand = si_rand_count(t3,  2)
print(si_t3_rand, '\n')
si_t3_sist = 0.4
si_t3 = np.round(np.sqrt(si_t3_rand**2 + si_t3_sist**2), 2)
print(si_t3, '\n')
T3 = np.round(np.mean(t3)/10, 2)
print('T3', T3, '\n')
si_T3 = si_t3/10

mc = 1617.8/1000     # ru
si_mc = 0.1/1000     # кг
dc = 78.1/1000       # м
si_dc = 0.1/1000     # м
Ic = np.round(mc * (dc/2)**2 / 2, 4)
print('Ic', Ic, '\n')
si_Ic = si_multi(Ic, si_dc, dc, 2, si_mc, mc, 1, dig_r = 6)
print(si_Ic, '\n')

Ir = np.round(Ic * (T3/T2)**2, 5)
print('Ir', Ir, '\n')
si_Ir = si_multi(Ir, si_Ic, Ic, 1, si_T3, T3, 2, si_T2, T2, 2, dig_r=5)
print('  ', si_Ir, '\n')


# In[12]:


wr = 1/k/Ir/2/3.14
print(wr, '\n')
si_wr = si_multi(wr, si_k, k, -1, si_Ir, Ir, -1)
print(si_wr, '\n')


# In[13]:


w0 = M[3]/Ir/w[3]/2/3.14
print(w0)


# In[203]:


print(M[::-1])

