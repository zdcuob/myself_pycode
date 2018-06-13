

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf

x = np.linspace(0,1,num=1024)
y = x
y = y / np.max(y)
yo = y.copy()

# two positive outliers
y[200:215] = y[200:215] + np.linspace(1,15, num=15) / 100
y[215:230] = y[215:230] + np.linspace(15,1, num=15) / 100
y[400:420] = y[400:420] + np.linspace(1,20,num=20) / 100
y[420:440] = y[420:440] + np.linspace(20,1,num=20) / 100


# two negative outliers
y[600:615] = y[600:615] - np.linspace(1,15, num=15) / 100
y[615:630] = y[615:630] - np.linspace(15,1, num=15) / 100
'''
y[800:820] = y[800:820] - np.linspace(1,20,num=20) / 100
y[820:840] = y[820:840] - np.linspace(20,1,num=20) / 100
'''

noise = 0.0001 * np.random.randn(1024)
y_n = y + noise     # with outliers 
yo_n = yo+ noise    # without outliers


plt.figure(1)
plt.subplot(121)
plt.plot(x,y_n)

plt.subplot(122)
plt.plot(x,yo_n)
plt.show()


# difference of y_n and yo_n
d_x = 1 / 1024
d_y=np.zeros(1023)
d_yo=np.zeros(1023)
for i in range(1023):
    d_y[i] = (y_n[i+1] - y_n[i]) / d_x
    d_yo[i] = (yo_n[i+1] - yo_n[i]) / d_x

# upper and lower thresholds of d_y    
y_up = np.mean(d_y) + 3 * np.std(d_y) * np.ones(1023)
y_lo = np.mean(d_y) - 3 * np.std(d_y) * np.ones(1023)

# upper and lower thresholds of d_yo
yo_up = np.mean(d_yo) + 3 * np.std(d_yo) * np.ones(1023)
yo_lo = np.mean(d_yo) - 3 * np.std(d_yo) * np.ones(1023)


def CHI(t,HI,L): 
# t is time
# HI is the difference of the original health index;
# L is the length of a outlier region
    if t.size != HI.size:
        print('The lengths of t and HI should be the same')

    num = HI.size
    d_t = (t[num-1] - t[0]) / num
    dHI = np.zeros(num-1)
    for i in range(num-1):     # diference of HI
        dHI[i] = (HI[i+1]- HI[i]) / d_t

    y_up = np.mean(dHI) + 3 * np.std(dHI)  # upper threshold 
    y_lo = np.mean(dHI) - 3 * np.std(dHI)  # lower threshold
    
    p_num = 0
    po = np.array([[],[]])

    for i in range(num-1): # search all outliers in HI and corresponding indexes 
        if dHI[i] > y_up:
            po = np.append(po,[[i],[HI[i]]],axis=1)
            p_num += 1
        elif dHI[i] < y_lo:
            po = np.append(po,[[i],[HI[i]]],axis=1)
            p_num += 1

    ind =  np.array([0]) # classify differnt zones 
    for i in range(p_num-1):
        if po[0,i+1] - po[0,i] >2:
            ind = np.append(ind,[i],axis=0)

    i_num = ind.size
    outlier =  np.array([[],[]])
    for i in range(i_num):
        outlier = np.append(outlier,po[:,ind[i]:ind[i+1]],axis=1)
        if outlier.all()>0 or outlier.all()>0

       
    return(ind)
'''
    for i in range(o_num/2-1):  
        if outlier[0,i+1] - outlier[0,i] > 2:
            ind = np.array([i])

#    i_num = ind.size
            
    return(ind)
'''
print(CHI(x,y_n,5))

plt.figure(2)
plt.subplot(121)
plt.plot(x[:1023],d_y,'b*', x[:1023], y_up, 'r--', x[:1023], y_lo, 'r--')

plt.subplot(122)
plt.plot(x[:1023],d_yo,'b*', x[:1023], yo_up, 'r--', x[:1023], yo_lo, 'r--')
plt.show()    
                







         
'''           




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

'''























