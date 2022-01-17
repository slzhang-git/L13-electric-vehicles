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

# TODO: Update and check the code to cater to each commodity
# Final Results
# Path Inflows
f = data['f']
# for p in f[np.newaxis][0].fPlus[0]:
    # f[np.newaxis][0].fPlus[0][p].drawGraph(0,50).show()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        ' ' , '']

# To plot different subplots in one figure
fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0]))
fig.suptitle('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
timeStep=%s \n\nPath inflows'%(insName,timeHorizon,maxIter,precision,alpha,\
timeStep))

for i,p in enumerate(f[()].fPlus[0]):
    x,y = f[()].fPlus[0][p].getXandY(0,20)
    a,b = [int(c) for c in x],[int(c) for c in y]
    print("i: ", i,a,b)
    # To plot in one figure together
    # plt.plot(x,y,label='path%d'%i, color=colors[i], linestyle=linestyles[i])

    # To plot different subplots in one figure
    axs[i].plot(x,y,label='path%d'%i, color=colors[i], linestyle=linestyles[i])
    axs[i].legend()
# plt.legend()
plt.xlabel('time')
plt.show()

# Travel Times
# To plot different subplots in one figure
# figt, axst = plt.subplots(len(f[np.newaxis][0].fPlus[0]))
# figt.suptitle('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
# timeStep=%s \nTravel Times'%(insName,timeHorizon,maxIter,precision,alpha,\
# timeStep))
tt = data['travelTime']
# print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
x = [float(timeStep)/2 + x*float(timeStep) for x in\
    range(int((float(timeHorizon)-0)/float(timeStep)))]
# print(timeHorizon, timeStep, x)
for i in range(len(tt)):
    for p in range(len(tt[i])):
        # To plot in one figure together
        y = tt[i][p]
        plt.plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])
        # To plot different subplots in one figure
        # axst[p].plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])
        # axst[p].legend()
plt.legend()
plt.title('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
timeStep=%s \n\nTravel Times'%(insName,timeHorizon,maxIter,precision,alpha,\
timeStep))
plt.xlabel('time')
plt.show()

# Alpha and FlowDiff per iteration
alphaIter = data['alphaIter']
diffBwFlowsIter = data['diffBwFlowsIter']
x = [x+1 for x in range(len(alphaIter))]
# To plot different subplots in one figure
fig1, axs1 = plt.subplots(2)

# To plot in one figure together
# plt.plot(x,alphaIter,label='alpha', color=colors[0], linestyle=linestyles[0])
# plt.plot(x,diffBwFlowsIter,label='flowDiff', color=colors[1], linestyle=linestyles[1])
# plt.plot([1,len(alphaIter)], [float(precision), float(precision)],label='precision',\
        # color=colors[2], linestyle=linestyles[2])
# To plot different subplots in one figure
axs1[0].plot(x,alphaIter,label='alpha', color=colors[0], linestyle=linestyles[0])
axs1[1].plot(x,diffBwFlowsIter,label='flowDiff', color=colors[1], linestyle=linestyles[1])
axs1[1].plot([1,len(alphaIter)], [float(precision), float(precision)],label='precision',\
        color=colors[2], linestyle=linestyles[2])
axs1[0].legend()
axs1[1].legend()
# plt.legend()
fig1.suptitle('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
timeStep=%s \n\nAlpha and FlowDiff'%(insName,timeHorizon,maxIter,precision,alpha,\
timeStep))
# plt.title('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
# timeStep=%s \nAlpha and FlowDiff'%(insName,timeHorizon,maxIter,precision,alpha,\
# timeStep))
plt.xlabel('iteration')
plt.show()



# for p in f.fPlus[0]:
    # f.fPlus[0][p].drawGraph(0,50).show()

# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()
# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()
# networkLoading(f).fPlus[G.edges[0], 0].drawGraph(0,50).show()


