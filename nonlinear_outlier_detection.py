

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf

x = np.linspace(0,1,num=1024)
y = np.exp(x)
y = y / np.max(y)
yo = y.copy()

# two positive outliers
y[200:215] = y[200:215] + np.linspace(1,15, num=15) / 100
y[215:230] = y[215:230] + 0.95 * np.linspace(15,1, num=15) / 100
y[400:420] = y[400:420] + np.linspace(1,20,num=20) / 100
y[420:440] = y[420:440] + np.linspace(20,1,num=20) / 100


# two negative outliers
y[600:615] = y[600:615] - np.linspace(1,15, num=15) / 100
y[615:630] = y[615:630] - np.linspace(15,1, num=15) / 100
'''
y[800:820] = y[800:820] - np.linspace(1,20,num=20) / 100
y[820:840] = y[820:840] - np.linspace(20,1,num=20) / 100
'''

noise = 0.0000 * np.random.randn(1024)
y_n = y + noise     # with outliers 
yo_n = yo+ noise    # without outliers


plt.figure(1)
plt.subplot(121)
plt.plot(x,y_n,'*')

plt.subplot(122)
plt.plot(x,yo_n,'*')
plt.show()

# difference of y
d_x = 1 / 1024
d_y=np.zeros(1023)
d_yo=np.zeros(1023)
for i in range(1023):
    d_y[i] = (y_n[i+1] - y_n[i]) / d_x
    d_yo[i] = (yo_n[i+1] - yo_n[i]) / d_x

# upper and lower thresholds of d_y    
y_up = (np.mean(d_y) + 3 * np.std(d_y)) * np.ones(1023)
y_lo = np.mean(d_y) - 3 * np.std(d_y) * np.ones(1023)

# upper and lower thresholds of d_yo
yo_up = np.mean(d_yo) + 3 * np.std(d_yo) * np.ones(1023)
yo_lo = np.mean(d_yo) - 3 * np.std(d_yo) * np.ones(1023)


plt.figure(2)
plt.subplot(121)
plt.plot(x[:1023],d_y,'b', x[:1023], y_up, 'r--', x[:1023], y_lo, 'r--')

plt.subplot(122)
plt.plot(x[:1023],d_yo,'b', x[:1023], yo_up, 'r--', x[:1023], yo_lo, 'r--')
plt.show()


def trend_1(t,HI): # trendability _ 1 
    t_m =np.mean(t)
    HI_m = np.mean(HI)
    num = HI.size
    a = np.zeros(num)
    b1 = np.zeros(num)
    b2 = np.zeros(num)
    
    for i in range(num):
        a[i] = (HI[i] - HI_m)*(t[i] - t_m)
        b1[i] = (HI[i] - HI_m)**2
        b2[i] = (t[i] - t_m)**2    
        
    a_num = np.abs(np.sum(a))
    b_den = np.sqrt(np.sum(b1)*np.sum(b2))
    tre = a_num / b_den
    return(tre)

print(trend_1(x,y_n))
print(trend_1(x,yo_n))


def trend_2(t,HI): # trendability _ 2 
    num = HI.size
    a_num = num * np.sum(t * HI) - np.sum(t) * np.sum(HI)
    b1_den = num * np.sum(t**2) -(np.sum(t))**2
    b2_den = num * np.sum(HI**2) -(np.sum(HI))**2
    tre = a_num / np.sqrt(b1_den*b2_den)
    return(tre)

print(trend_2(x,y_n))
print(trend_2(x,yo_n))


def monot(HI): # monotonicity
    num = HI.size
    a = np.zeros(num)
    b = np.zeros(num)
    for i in range(num-1):
        a[i] = np.heaviside(HI[i+1] - HI[i],0)
        b[i] = np.heaviside(HI[i] - HI[i+1],0)

    mono = np.abs(np.sum(a)-np.sum(b)) / (num-1)
    return mono

print(monot(y_n))
print(monot(yo_n))


def rob(t,HI): # robustness
    num = HI.size 
    z = np.polyfit(t, HI, 2)  # 2-order polynomial fitting 
    p = np.poly1d(z)
    yval = p(t)
    a = np.zeros(num)
    for i in range(num):
        a[i] = (HI[i] - yval[i])/HI[i]
    robust = np.sum(np.exp(-np.abs(a))) / num
    return robust

print(rob(x,y_n))
print(rob(x,yo_n))

























