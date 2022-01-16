from __future__ import annotations

import heapq, math
from typing import List, Dict

from network import *
from flows import *
from utilities import *

from networkloading import *
from fixedPointAlgorithm import *
# import os, gzip
import time, sys
import numpy as np
import matplotlib.pyplot as plt

# TODO: Use pickle to store class objects
data = np.load(sys.argv[1], allow_pickle=True);
print(data.files)
# f
f = data['f']

print("f: ", f, type(f), f.size, f.shape, f[()])
print("fPlus: ")
# for p in f[()].fPlus[0]:
    # f[()].fPlus[0][p].drawGraph(0,50).show()
# exit(0)
# print("attr: ", f[np.newaxis])
# print("attr: ", f[np.newaxis][0].fPlus[0])

# Eventual Flow
# eventualFlow=data['eventualFlow']
# print("eventualFlow: ", eventualFlow)
# print("attrs: ", eventualFlow[np.newaxis][0].fPlus)
# for p in f[np.newaxis][0].fPlus[0]:
    # f[np.newaxis][0].fPlus[0][p].drawGraph(0,50).show()
realf = f[np.newaxis][0].fPlus[0]
realf = f[np.newaxis][0].fPlus[0]
print(realf)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        ' ' , '']

fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0]))
fig.suptitle('Path flows')

for i,p in enumerate(f[()].fPlus[0]):
    x,y = [],[]
    x,y = f[()].fPlus[0][p].drawGraph(0,20)[1:3]
    # plt.plot(x,y,label='p%d'%i, color=colors[i], linestyle=linestyles[i])
    a = [int(c) for c in x]
    b = [int(c) for c in y]
    print("i: ", i,a,b)
    axs[i].plot(x,y,label='p%d'%i, color=colors[i], linestyle=linestyles[i])
    axs[i].legend()
# plt.legend(['p1','p2','p3'])
# plt.legend()
plt.show()


# for p in f.fPlus[0]:
    # f.fPlus[0][p].drawGraph(0,50).show()

# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()
# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()
# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()


