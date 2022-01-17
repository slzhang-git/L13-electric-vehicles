from __future__ import annotations

import heapq, math
from typing import List, Dict

from network import *
from flows import *
from utilities import *

from networkloading import *
from fixedPointAlgorithm import *
import os
import time, sys
import numpy as np
import matplotlib.pyplot as plt
import re

# TODO: Use pickle to store class objects
data = np.load(sys.argv[1], allow_pickle=True);
fname = os.path.splitext(os.path.split(sys.argv[1])[1])[0]
[insName,timeHorizon,maxIter,precision,alpha,timeStep] = re.split('[_]', fname)
print(data.files)


# print("f: ", f, type(f), f.size, f.shape, f[()])
# print("fPlus: ")

# Final Results
# Path Inflows
f = data['f']
# for p in f[np.newaxis][0].fPlus[0]:
    # f[np.newaxis][0].fPlus[0][p].drawGraph(0,50).show()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        ' ' , '']

# To plot in different subplots in one figure
fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0]))
fig.suptitle('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
timeStep=%s \nPath inflows'%(insName,timeHorizon,maxIter,precision,alpha,\
timeStep))

for i,p in enumerate(f[()].fPlus[0]):
    x,y = f[()].fPlus[0][p].getXandY(0,20)
    # a,b = [int(c) for c in x],[int(c) for c in y]
    # print("i: ", i,a,b)
    # To plot in one figure together
    # plt.plot(x,y,label='p%d'%i, color=colors[i], linestyle=linestyles[i])

    # To plot in different subplots in one figure
    axs[i].plot(x,y,label='p%d'%i, color=colors[i], linestyle=linestyles[i])
    axs[i].legend()
# plt.legend()
plt.show()


# for p in f.fPlus[0]:
    # f.fPlus[0][p].drawGraph(0,50).show()

# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()
# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()
# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()


