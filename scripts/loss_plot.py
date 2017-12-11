#Affichage Loss

import matplotlib.pyplot as plt
from constants import *


f=open(DIR_TO_SAVE+"/loss_memory.txt",'r')

l=f.readlines()

n_update=[]
tp=[]
d=[]
g=[]
e=[]



for k in l:
    m=k.split()
    if m!=[]:
        n_update.append(m[0])
        tp.append(m[1])
        d.append(m[2])
        g.append(m[3])
        e.append(m[4])


#plt.plot(n_update,tp,'black')

#plt.figure()
plt.plot(n_update,d,'red')
plt.title("D_obj = binary_crossentropy on discriminator")

plt.figure()
plt.plot(n_update,g,'blue')
plt.title("G_obj = binary_crossentropy on generator + train_err * alpha")

plt.figure()
plt.plot(n_update,e,'green')
plt.title("train_err = binary_crossentropy on the whole Network")

plt.show()
