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
[insName,timeHorizon,maxIter,precision,alpha,timeStep,_] = re.split('[_]', fname)
runTime = round(float(data['time']),2)
print("Data: ", data.files)
print("Termination message: ", data['stopStr'])
print("Time taken: ", runTime)
print("Iterations: ", len(data['alphaIter']))

# print(fname,insName,timeHorizon,maxIter,precision,alpha,timeStep)
# print("f: ", f, type(f), f.size, f.shape, f[()])
# print("fPlus: ")

# TODO: Update and check the code to cater to each commodity
# Final Results
#-----------------
# Path Inflows
#-----------------
f = data['f']
# for p in f[np.newaxis][0].fPlus[0]:
    # f[np.newaxis][0].fPlus[0][p].drawGraph(0,50).show()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        ' ' , '']
locs = ["upper left", "lower left", "center left", "center right"]
# To plot different subplots in one figure
# TODO: Find the number of paths that have positive inflows etc. to find the exact
# number of subplots required
# fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0]) + 1)
fig, axs = plt.subplots(3)
# fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0])+1+2, figsize=(11.7,  8.3))
# fig_width, fig_height = plt.gcf().get_size_inches()
# print(fig_width, fig_height)

# Set fontsize
fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
params = {'legend.fontsize': 'x-large',
        # 'figure.figsize': (15, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize':'xx-large',
        'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
# print(plt.rcParams.keys())
# plt.rc('legend',fontsize=20) # using a size in points
# plt.rc('legend',fontsize=fontsizes[5]) # using a named size
alphaStr = data['alphaStr']
fig.suptitle(r'ins=[%s], $T$=[%s], $maxIter$=[%s], $\epsilon$=[%s], $\alpha_0$=[%s],'\
# r'timeStep=%s, $\alpha-$update: %s''\n\nPath inflows'%(insName,timeHorizon,maxIter,precision,alpha,\
r' timeStep=[%s], $\alpha-$update rule: [%s]''\n runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
timeStep,alphaStr,runTime), fontsize='xx-large')

k = -1
k += 1
for i,p in enumerate(f[()].fPlus[0]):
    # k += 1
    x,y = f[()].fPlus[0][p].getXandY(0,20)
    # a,b = [int(c) for c in x],[int(c) for c in y]
    # print("i: ", i,a,b)
    # To plot in one figure together
    # plt.plot(x,y,label='path%d'%i, color=colors[i], linestyle=linestyles[i])

    # To plot different subplots in one figure
    # if max(y)>0:
    axs[k].plot(x,y,label='path%d'%i, color=colors[i], linestyle=linestyles[i])
    # axs[k].legend()
    # else:
        # k -= 1
# plt.legend()
# plt.xlabel('time')
# plt.show()
axs[k].legend()
axs[k].set_title('Path Inflows', fontsize='xx-large')
# axs[2].set_ylim(bottom=-0.01)
#-----------------
# Travel Times
#-----------------
# To plot different subplots in one figure
# figt, axst = plt.subplots(len(f[np.newaxis][0].fPlus[0]))
# figt.suptitle('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
# timeStep=%s \nTravel Times'%(insName,timeHorizon,maxIter,precision,alpha,\
# timeStep))
tt = data['travelTime']
# print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
# x = [float(timeStep)/2 + x*float(timeStep) for x in\
    # range(int((float(timeHorizon)-0)/float(timeStep)))]
x = [float(timeStep)/2 + x*float(timeStep) for x in\
    range(int((len(tt[0][0])-0)))]
# print(timeHorizon, timeStep, x)
for i in range(len(tt)):
    k += 1
    for p in range(len(tt[i])):
        # To plot in one figure together
        y = tt[i][p]
        # plt.plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])
        axs[k].plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])
        # To plot different subplots in one figure
        # axst[p].plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])
        axs[k].legend()
# plt.legend()
# plt.title('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
# timeStep=%s \n\nTravel Times'%(insName,timeHorizon,maxIter,precision,alpha,\
# timeStep))
axs[k].set_xlabel('time', fontsize='xx-large')
axs[k].set_title('Travel Times', fontsize='xx-large')
# plt.show()

#-----------------
# Alpha and FlowDiff per iteration
#-----------------
alphaIter = data['alphaIter']
diffBwFlowsIter = data['diffBwFlowsIter']
# a,b = [round(float(c),2) for c in alphaIter],[round(float(c),2) for c in diffBwFlowsIter]
# print(a,b)
x = [x+1 for x in range(len(alphaIter))]
# To plot different subplots in one figure
# fig1, axs1 = plt.subplots(2)

# To plot in one figure together
# plt.plot(x,alphaIter,label='alpha', color=colors[0], linestyle=linestyles[0])
# plt.plot(x,diffBwFlowsIter,label='flowDiff', color=colors[1], linestyle=linestyles[1])
# plt.plot([1,len(alphaIter)], [float(precision), float(precision)],label='precision',\
        # color=colors[2], linestyle=linestyles[2])
# To plot different subplots in one figure
k += 1
axs[k].plot(x,alphaIter,label=r'$\alpha$', color=colors[0], linestyle=linestyles[0])
# axs[k].legend(loc=locs[3])
# k += 1
axs2 = axs[k].twinx()
axs2.plot(x,diffBwFlowsIter,label=r'$\Delta$ f', color=colors[1], linestyle=linestyles[1])
axs2.plot([1,len(alphaIter)], [float(precision), float(precision)],label=r'$\epsilon$',\
        color=colors[2], linestyle=linestyles[2])
axs2.legend(loc=locs[3], fontsize='x-large')
# plt.legend()
# fig1.suptitle('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
# timeStep=%s \n\nAlpha and FlowDiff'%(insName,timeHorizon,maxIter,precision,alpha,\
# timeStep))
# plt.title('insName=%s timeHorizon=%s maxIter=%s precision=%s alpha=%s \
# timeStep=%s \nAlpha and FlowDiff'%(insName,timeHorizon,maxIter,precision,alpha,\
# timeStep))
axs[k].set_xlabel('iteration', fontsize='xx-large')
axs[k].set_title(r'$\alpha$ and $\Delta f^{k}$', fontsize='xx-large')
# plt.show()

# Set label and xtick sizes for axes
for i in range(len(axs)):
    axs[i].legend(fontsize='x-large')
    # axs[i].set_xticklabels('None', fontsize='x-large')
    # axs[i].xaxis.get_major_ticks().label.set_fontsize('x-small')
    plt.setp(axs[i].get_xticklabels(), fontsize='x-large')
    plt.setp(axs[i].get_yticklabels(), fontsize='x-large')
    # axs[i].set_xlabel('xlabel', fontsize='x-large')

# TODO: Avoid this hardcoding
axs[2].legend(loc=locs[2])

plt.ylim(bottom=0)

# plt.draw()
# axs.apply_aspect()
# fig.tight_layout()
mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
mng.full_screen_toggle()
plt.show()
# plt.show(block=False)
# plt.pause(.1)
plt.close()

# Save figure
dirname = os.path.expanduser('./figures')
figname = os.path.join(dirname, fname)
# plt.savefig(figname, bbox_inches='tight')
fig.savefig(figname, format='png', dpi=fig.dpi, bbox_inches='tight')
