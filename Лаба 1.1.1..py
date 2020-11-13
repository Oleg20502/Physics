#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math as m
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


nd = 10
si_l = 0.1  #cm
D2 = [0.37, 0.37, 0.37, 0.37, 0.36, 0.36, 0.36, 0.37, 0.37, 0.36]
nd = len(D2)
print(nd)


# In[4]:


d1m = 0.35
d2m = round(sum(D2)/nd, 3)
si_si = 0.01 #mm
si_rand = 0  #пока
print(d2m)


# In[5]:


a = 0
for i in range(nd):
    a += (D2[i] - d2m)**2
si_rand = round(m.sqrt(a)/nd, 4)
si_d = round(m.sqrt(si_si**2 + si_rand**2), 4)
print(si_rand)
print(si_d)


# In[6]:


S = round(3.14 * d2m**2/4/100, 6)
print(S)  #cm^2


# In[7]:


si_s = round(2*si_d/d2m*S, 3)
print(si_s)


# In[8]:


p1 = round(si_d/d2m*100, 2)  #точность d
print(p1)
p2 = round(si_s/S, 2)  #точность S
print(p2)


# In[9]:


l1 = 20 #cm
u1 = np.array([133, 101, 82, 62, 50, 42, 57, 71, 90, 111])
i1 = np.array([327.6, 249.1, 202.5, 153.2, 123.5, 104.6, 140.0, 174.7, 222.6, 274.2])

l2 = 30 #cm
u2 = np.array([138, 110, 91, 80, 70, 60, 75, 85, 101, 121])
i2 = np.array([225.9, 180.5, 149.2, 131.3, 114.9, 98.5, 123.4, 139.2, 166.2, 197.9])

l3 = 50 #cm
u3 = np.array([132, 121, 105, 92, 82, 77, 86, 99, 113, 128])
i3 = np.array([129.2, 118.1, 103.3, 89.9, 80.2, 75.6, 84.4, 96.8, 110.3, 125.3])

exp = 2

if exp == 1:
    v1 = u1
    I1 = i1
    l = l1
if exp == 2:
    v1 = u2
    I1 = i2
    l = l2
if exp == 3:
    v1 = u3
    I1 = i3
    l = l3


# In[10]:


V1 = v1 * 5
n = len(V1)
m1, m2 = 0,0
print(V1)


# In[11]:


Rm = round(np.sum(V1*I1)/np.sum(I1**2), 3)
print(Rm)
Rpr = round(Rm*(1+Rm/5000), 3)
print(Rpr)


# In[12]:


si_r_rand = round(np.sqrt((np.sum(V1**2)/np.sum(I1**2) - Rm**2)/n), 3)
print(si_r_rand)
si_v = 1.875/2
si_i = 0.002*np.max(I1) + 0.02
si_r_si = round(Rm*np.sqrt((si_v/max(V1))**2+(si_i/max(I1)**2)), 3)
print(si_r_si)
si_r = round(np.sqrt(si_r_rand**2 + si_r_si**2), 3)
print(si_r)
#print(np.sum(V1**2))
#print(np.sum(I1**2))
#print((np.sum(V1**2)/round(np.sum(I1**2), 3) - Rm**2))


# In[13]:


plt.plot(I1, V1)
plt.show()


# In[14]:


po = round(Rpr*S/l, 7)
print(po)


# In[15]:


si_po = round(po*np.sqrt((si_r/Rpr)**2 + (2*si_d/d2m)**2 + (si_l/l)**2), 6)
print(si_po)


# In[20]:


#import matplotlib.pyplot as plt
x, y = np.sort(I1), np.sort(V1)
plt.errorbar(x, y, xerr=si_i, yerr=si_v)
y = Rm*x
plt.plot(x,y)
plt.grid()
plt.show()


# In[21]:


p, v = np.polyfit(x, y, deg=1, cov=True)
p_f = np.poly1d(p)
plt.plot(x, p_f(x))
plt.grid()
plt.show()
print(p, v)


# In[55]:


x = np.arange(0, 10, 0.01)
plt.plot(x, x**2, label=r'$f = x^2$')
plt.scatter(x, x**2 + np.random.randn(len(x))*x, s=0.3)
plt.fill_between(x, 1.3*x**2, 0.7*x**2, alpha=0.6)
plt.legend(loc='best')
plt.show()


# In[ ]:




