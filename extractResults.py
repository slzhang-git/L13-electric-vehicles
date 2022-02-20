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
print(re.split('[_]', fname))
[insName,timeHorizon,maxIter,precision,alpha,timeStep,_,_] = re.split('[_]', fname)
runTime = round(float(data['time']),2)
print("Data: ", data.files)
print("Termination message: ", data['stopStr'])
print("Time taken: ", runTime)
print("Iterations: ", len(data['alphaIter']))

# print(fname,insName,timeHorizon,maxIter,precision,alpha,timeStep)
# print("f: ", f, type(f), f.size, f.shape, f[()])
# print("fPlus: ")

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange',
        'darkviolet','magenta','darkorchid','darkgreen']
# linestyles = ['solid', 'dashed', 'dashdot', 'dotted', 'offset',\
        # 'on-off-dash-seq', '-' , '--' , '-.' , ':' , 'None' ,\
        # ' ' , '']
fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-', '--',\
        '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot',\
        'dotted']
locs = ["upper left", "lower left", "center left", "center right", "upper right"]

f = data['f']

# TODO: Find the number of paths that have positive inflows etc. to find the exact
# number of subplots (if required)
# fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0]) + 1)
print(f[()].fPlus)
for c,p in enumerate(f[()].fPlus):
    # print('comm:%d'%c, f[()].fPlus[c], f[()].getEndOfInflow(c))
    # exit(0)
    print('Commodity: %d'%c)

    # Making figures for each commodity
    fig, axs = plt.subplots(4)

    # Set fontsize
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
    # alphaStr = data['alphaStr']

    # TODO: Update and check the code to cater to each commodity
    # Final Results
    #-----------------
    # Path Inflows
    #-----------------
    fig.suptitle(r'ins=[%s], $T$=[%s], $maxIter$=[%s], $\epsilon$=[%s], $\alpha_0$=[%s],'\
    # r' timeStep=[%s], $\alpha-$update rule: [%s]''\n runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
    # timeStep,alphaStr,runTime), fontsize='xx-large')
    r' timeStep=[%s], runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
    timeStep,runTime), fontsize='xx-large')

    k = -1
    k += 1
    for i,p in enumerate(f[()].fPlus[0]):
        # k += 1
        #TODO: Determine the right end of x-axis for plots
        # x,y = f[()].fPlus[0][p].getXandY(0,20)
        x,y = f[()].fPlus[0][p].getXandY(0,f[()].getEndOfInflow(c))
        # a,b = [int(c) for c in x],[int(c) for c in y]
        # print("i: ", i,a,b)
        # if max(y)>0:
        axs[k].plot(x,y,label='path%d'%i, color=colors[i], linestyle=linestyles[i])
        # axs[k].plot(x,y,label='path%d'%i, linestyle=linestyles[i])
        # axs[k].legend()
        # else:
            # k -= 1
    axs[k].legend(loc='upper right')
    axs[k].set_title('Path Inflows', fontsize='xx-large')
    # plt.show()

    #-----------------
    # Travel Times
    #-----------------
    tt = data['travelTime']
    # print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
    print(tt[c], len(tt[c]), len(tt[c][0]))
    x = [float(timeStep)/2 + x*float(timeStep) for x in\
        range(int((len(tt[0][0])-0)))]
    # print(timeHorizon, timeStep, x)
    k += 1
    # for i in range(len(tt)):
        # k += 1
    for p in range(len(tt[c])):
        y = tt[c][p]
        axs[k].plot(x,y,label='path%d'%p, color=colors[p], linestyle=linestyles[p])
        # axs[k].plot(x,y,label='path%d'%p, linestyle=linestyles[p])
    axs[k].legend(loc='upper right')
    axs[k].set_xlabel('time', fontsize='xx-large')
    axs[k].set_title('Travel Times', fontsize='xx-large')
    # plt.show()

    #-----------------
    # Alpha and FlowDiff per iteration
    #-----------------
    alphaIter = data['alphaIter']
    absDiffBwFlowsIter = data['absDiffBwFlowsIter']
    relDiffBwFlowsIter = data['relDiffBwFlowsIter']
    qopiIter = data['qopiIter']
    qopiMeanIter = data['qopiMeanIter']
    # a,b = [round(float(c),2) for c in alphaIter],[round(float(c),2) for c in diffBwFlowsIter]
    # print(a,b)
    x = [x+1 for x in range(len(alphaIter))]

    k += 1
    axs[k].plot(x,alphaIter,label=r'$\alpha$', color=colors[0], linestyle=linestyles[0])
    axs2 = axs[k].twinx()
    axs2.plot(x,absDiffBwFlowsIter,label=r'$\Delta$ f', color=colors[1], linestyle=linestyles[1])
    axs2.plot(x,relDiffBwFlowsIter,label=r'($\Delta$ f / f)', color=colors[2], linestyle=linestyles[1])
    axs2.plot([1,len(alphaIter)], [float(precision), float(precision)],label=r'$\epsilon$',\
            color=colors[2], linestyle=linestyles[2])
    axs2.legend(loc=locs[3], fontsize='x-large')
    # axs[k].set_xlabel('iteration', fontsize='xx-large')
    # axs[k].set_title(r'$\alpha$ and $\Delta f^{k}$', fontsize='xx-large')
    # plt.show()
   # TODO: Avoid this hardcoding
    # axs[2].legend(loc=locs[2])
    axs[k].legend(loc=locs[1], fontsize='x-large')

    k += 1
    axs[k].plot(x,qopiIter,label='QoPI', color=colors[3], linestyle=linestyles[1])
    axs[k].plot(x,qopiMeanIter,label='QoPIMean', color=colors[4], linestyle=linestyles[1])
    axs[k].set_xlabel('iteration', fontsize='xx-large')
    axs[k].legend(fontsize='x-large')

    # Set label and xtick sizes for axes
    for i in range(len(axs)):
        # axs[i].legend(fontsize='x-large')
        plt.setp(axs[i].get_xticklabels(), fontsize='x-large')
        plt.setp(axs[i].get_yticklabels(), fontsize='x-large')

    plt.ylim(bottom=0)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    # plt.show(block=False)
    # plt.pause(.1)
    plt.close()

    # Save figure
    dirname = os.path.expanduser('./figures')
    fname1 = fname + '_comm%d'%c
    print(fname)
    figname = os.path.join(dirname, fname1)
    fig.savefig(figname, format='png', dpi=fig.dpi, bbox_inches='tight')
    # exit(0)
