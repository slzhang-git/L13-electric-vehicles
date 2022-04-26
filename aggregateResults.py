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
# for axhline
import matplotlib.transforms as transforms

# TODO: Use pickle to store class objects
data = np.load(sys.argv[1], allow_pickle=True);
fname = os.path.splitext(os.path.split(sys.argv[1])[1])[0]
print(re.split('[_]', fname), len(re.split('[_]', fname)))
if len(re.split('[_]', fname)) > 9:
    [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,priceToTime,numThreads,alphaStr] = re.split('[_]', fname)
else:
    if len(re.split('[_]', fname)) > 8:
        [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,numThreads,alphaStr] = re.split('[_]', fname)
    else:
        [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,alphaStr] = re.split('[_]', fname)
runTime = round(float(data['time']),2)
print("Data: ", data.files)
print("Termination message: ", data['stopStr'])
print("Time taken: ", runTime)
print("Iterations: ", len(data['alphaIter']))

commodities = data['commodities']
energyBudget = commodities[0][2]

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

G = data['G']
f = data['f']

# Reading another data file
# data1 = np.load(sys.argv[2], allow_pickle=True);
# f1 = data1['f']
# fname1 = os.path.splitext(os.path.split(sys.argv[2])[1])[0]
# print(re.split('[_]', fname1))
# [insName1,timeHorizon1,maxIter1,timeLimit1,precision1,alpha1,timeStep1,alphaStr1] = re.split('[_]', fname1)

# TODO: Find the number of paths that have positive inflows etc. to find the exact
# number of subplots (if required)
# fig, axs = plt.subplots(len(f[np.newaxis][0].fPlus[0]) + 1)
# print(f[()].fPlus[0])
# for c,p in enumerate(f[()].fPlus):
    # # print('comm:%d'%c, f[()].fPlus[c], f[()].getEndOfInflow(c))
    # # exit(0)
    # print('Commodity: %d'%c)

    # # Making figures for each commodity
    # fig, axs = plt.subplots(4)

    # # Set fontsize
    # params = {'legend.fontsize': 'x-large',
            # # 'figure.figsize': (15, 5),
            # 'axes.labelsize': 'xx-large',
            # 'axes.titlesize': 'xx-large',
            # 'xtick.labelsize':'xx-large',
            # 'ytick.labelsize':'x-large'}
    # plt.rcParams.update(params)
    # alphaStr = data['alphaStr']

    # # TODO: Update and check the code to cater to each commodity
    # # Final Results
    # #-----------------
    # # Walk Inflows
    # #-----------------
    # fig.suptitle(r'ins=[%s], $T$=[%s], $maxIter$=[%s], $\epsilon$=[%s], $\alpha_0$=[%s],'\
    # r' timeStep=[%s], $\alpha-$update rule: [%s]''\n runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
    # timeStep,alphaStr,runTime), fontsize='xx-large')
    # # r' timeStep=[%s], runTime=%.2f'%(insName,timeHorizon,maxIter,precision,alpha,\
    # # timeStep,runTime), fontsize='xx-large')

    # k = -1
    # k += 1
    # figF, axsF = plt.subplots(1)
    # figB, axsB = plt.subplots(1)
    # u = sum([f[()].fPlus[c][p1].integrate(0,1) for p1 in f[()].fPlus[c]])
    # # bmax = 1 + max([2*p.getNetEnergyConsump() for p in f[()].fPlus[c]])
    # bmax =13
    # bmin = min([p.getNetEnergyConsump() for p in f[()].fPlus[c]])
    # # for p in f[()].fPlus[c]:
        # # print(p, p.getNetEnergyConsump())
    # # print('bmin ',bmin)
    # yBsum = []
    # yBsum1 = []
    # tt = data['travelTime']
    # xB1 = [float(timeStep)/2 + x*float(timeStep) for x in range(int((len(tt[0][0])-0)))]

    # # for p in f[()].fPlus[c]:
        # # fmax += f[()].fPlus[c][p].integrate(0,1)
    # for i,p1 in enumerate(f[()].fPlus[c]):
        # # k += 1
        # #TODO: Determine the right end of x-axis for plots
        # # x,y = f[()].fPlus[0][p].getXandY(0,20)
        # x,y = f[()].fPlus[c][p1].getXandY(0,f[()].getEndOfInflow(c))
        # yB = [p1.getNetEnergyConsump()*v/u for v in y]
        # y1 = [f[()].fPlus[c][p1].getValueAt(i) for i in xB1]
        # yB1 = [p1.getNetEnergyConsump()*v/u for v in y1]
        # if (len(y) > 2):
            # xB = x
            # # [lambda: value_false, lambda: value_true][<test>]()
            # # yBsum = [lambda:yB, lambda:[yBsum[i]+yB[i] for i,_ in enumerate(yB)]][len(yBsum)>0]()
            # yBsum1 = [lambda:yB1, lambda:[yBsum1[i]+yB1[i] for i,_ in enumerate(yB1)]][len(yBsum1)>0]()
            # print('i', i, p1.getNetEnergyConsump(), len(yB))
            # print(yB)
            # # axsB.plot(x,yB,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1], linewidth=10)
        # # print('i', i)
        # # print('y', len(y), y, p1.getNetEnergyConsump())
        # # print('yB', len(yB), yB)
        # # print('yBsum', len(yBsum), yBsum)

        # # a,b = [int(c) for c in x],[int(c) for c in y]
        # # print("i: ", i,a,b)
        # # if max(y)>0:
        # axs[k].plot(x,y,label='w%d'%i, color=colors[i], linestyle=linestyles[i])
        # # axs[k].plot(x,y,label='w%d'%i, linestyle=linestyles[i])
        # # axs[k].legend()
        # # else:
            # # k -= 1
        # axsF.plot(x,y,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1],
                # linewidth=10)
        # # axsB.plot(x,yB,label='Total', color=colors[i], linestyle=linestyles[1],
                # # linewidth=10)
        # # Temporary: uncomment for cases 2 and 3
        # # axsF.plot(x,y,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1],
                # # linewidth=10)
    # print('yBsum', yBsum)
    # # exit(0)
    # # axsB.plot(xB,yBsum,label='Total', color='cyan', linestyle=linestyles[2],
            # # linewidth=10)
    # # axsB.plot(xB,yBsum,label='with recharging', color='darkgreen', linestyle=linestyles[1],
            # # linewidth=10)
    # axsB.plot(xB1,yBsum1,label='with recharging', color='darkgreen', linestyle=linestyles[1],
            # linewidth=10)
    # # axsB.plot(xB,[float(energyBudget) for i in xB], label=r'$b^{max}$', color=colors[-2], linestyle='dotted',
            # # linewidth=10)
    # # axsB.plot(xB,[float(bmin) for i in xB], label=r'$b_{min}$', color='orange', linestyle='dotted',
            # # linewidth=10)
    # axs[k].legend(loc='upper right')
    # axs[k].set_title('Walk Inflows', fontsize='xx-large')
    # # plt.show()
    # axsF.legend(loc='best', fontsize=80, frameon=False, ncol=2)
    # axsF.set_xlabel(r'time ($\theta$)', fontsize=80)
    # axsB.legend(loc='best', fontsize=80, frameon=False)
    # # axsB.legend(loc='best', fontsize=80, frameon=False, ncol=2)
    # axsB.set_xlabel(r'time ($\theta$)', fontsize=80)
    
    # # Temporary: uncomment if y-ticks and y-labels are not needed
    # axsF.set_ylabel(r'Walk Inflows ($h^+$)', fontsize=80)
    # axsB.set_ylabel(r'Battery Consump. / Unit Flow', fontsize=80)
    # # axsF.set_yticks([])
    # # axsF.set_yticklabels([])

    # axsF.set_ylim([0, fmax])
    # axsB.set_ylim([0, bmax])

    # # axsB.set_yticks([])
    # # axsB.set_yticklabels([])
    
    # plt.setp(axsF.get_xticklabels(), fontsize=80)
    # plt.setp(axsF.get_yticklabels(), fontsize=80)

    # plt.setp(axsB.get_xticklabels(), fontsize=80)
    # plt.setp(axsB.get_yticklabels(), fontsize=80)

    # ### ANOTHER ENERGY PROFILE
    # # yBsum = []
    # # for i,p1 in enumerate(f1[()].fPlus[c]):
        # # x,y = f1[()].fPlus[c][p1].getXandY(0,f[()].getEndOfInflow(c))
        # # yB = [p1.getNetEnergyConsump()*v/u for v in y]
        # # if (len(y) > 2):
            # # xB = x
            # # # [lambda: value_false, lambda: value_true][<test>]()
            # # yBsum = [lambda:yB, lambda:[yBsum[i]+yB[i] for i,_ in enumerate(yB)]][len(yBsum)>0]()
            # # print('i', i, p1.getNetEnergyConsump())
            # # print(yB)
            # # # axsB.plot(x,yB,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1], linewidth=10)
    # # print('yBsum', yBsum)
    # # axsB.plot(xB,yBsum,label='without recharging', color='cyan', linestyle=linestyles[1],
            # # linewidth=10)
    # # axsB.plot(xB,[float(energyBudget) for i in xB], label=r'$b^{max}$', color=colors[-2], linestyle='dotted',
            # # linewidth=10)
    # # axsB.legend(loc='best', fontsize=80, frameon=False)
    # # axsB.set_xlabel(r'time ($\theta$)', fontsize=80)
    
    # # axsB.set_ylabel(r'Battery Consump. / Unit Flow', fontsize=80)
    # # axsB.set_ylim([0, bmax])
    
    # # plt.setp(axsB.get_xticklabels(), fontsize=80)
    # # plt.setp(axsB.get_yticklabels(), fontsize=80)

    # ###############


    # #------------------------
    # # Travel Times, QoPI_Walk
    # #------------------------
    # tt = data['travelTime']
    # qopi = data['qopiPathComm']
    # # print(tt)
    # # print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
    # # print(tt[c], len(tt[c]), len(tt[c][0]))

    # ## Temporary: Uncomment for permuting rows for consistent paths colors for case 3
    # # permutation = [0, 3, 4, 1, 2, 5, 6]
    # # idx = np.empty_like(permutation)
    # # idx[permutation] = np.arange(len(permutation))
    # # tt = tt[:, idx]
    # # print(tt)
    # ###############
    # x = [float(timeStep)/2 + x*float(timeStep) for x in\
        # range(int((len(tt[0][0])-0)))]
    # # print(timeHorizon, timeStep, x)
    # k += 1
    # # for i in range(len(tt)):
        # # k += 1
    # figtt, axsT = plt.subplots(1)
    # figTC, axsTC = plt.subplots(1)
    # figQ, axsQ = plt.subplots(1)
    # ttmax = np.amax(tt[c]) + 1
    # qmax = np.amax(qopi[c]) + np.amin(qopi[c][np.nonzero(qopi[c])])
    # qmax = 0.001

    # tmin = []
    # # tmax = []
    # # tmean = []
    # for p in range(len(tt[c])):
        # y = tt[c][p]
        # yQ = qopi[c][p]
        # # print(y)
        # axs[k].plot(x,y,label='w%d'%p, color=colors[p], linestyle=linestyles[p])

        # # [lambda: value_false, lambda: value_true][<test>]()
        # tmin = [lambda:y, lambda:[min(tmin[i],y[i]) for i,_ in enumerate(y)]][p>0]()
        # # tmax = [lambda:y, lambda:[max(tmax[i],y[i]) for i,_ in enumerate(y)]][p>0]()
        
        # # Temporary: uncomment for cases 2 and 3
        # # axsT.plot(x,y,label='w%d'%(p+1), color=colors[p+1], linestyle=linestyles[1],
                # # linewidth=10)
        # # axsQ.plot(x,yQ,label='w%d'%(p+1), color=colors[p+1], linestyle=linestyles[1],
                # # linewidth=10)
        
        # axsT.set_ylim([0, ttmax])
        # axsQ.set_ylim([0, qmax])
        # # axs[k].plot(x,y,label='w%d'%p, linestyle=linestyles[p])
        # print('time%d'%p, y)
    
    # axs[k].legend(loc='upper right')
    # axs[k].set_xlabel('time', fontsize='xx-large')
    # axs[k].set_title('Travel Times', fontsize='xx-large')
    # # plt.show()
    
    # axsT.legend(loc='best', fontsize=80, frameon=False, ncol=2)
    # axsT.set_xlabel(r'time ($\theta$)', fontsize=80)
    # # Temporary: uncomment if y-ticks and y-labels are not needed
    # # axsT.set_ylabel(r'travel time ($\psi$)', fontsize=80)
    # axsT.set_yticks([])
    # axsT.set_yticklabels([])

    # # Temporary: uncomment if y-ticks and y-labels are not needed
    # axsQ.legend(loc='best', fontsize=80, frameon=False, ncol=2)
    # axsQ.set_xlabel(r'time ($\theta$)', fontsize=80)
    # # axsQ.set_ylabel(r'QoPI', fontsize=80)
    # # axsQ.set_yticks([])
    # # axsQ.set_yticklabels([])

    # plt.setp(axsT.get_xticklabels(), fontsize=80)
    # plt.setp(axsT.get_yticklabels(), fontsize=80)
    # plt.setp(axsQ.get_xticklabels(), fontsize=80)
    # plt.setp(axsQ.get_yticklabels(), fontsize=80)
    # # axsT.set_title('Travel Times', fontsize='xx-large')

    # ## PLOT FOR MINIMUM, MAXIMUM and MEAN TRAVEL TIMES
    # print('\ntmin', tmin)
    # print('x', x)
    # # print('len tmin', len(tmin), len(x))
    # axsTC.plot(x,tmin,label='with recharging', color='darkgreen', linestyle=linestyles[1],
            # linewidth=10)
    # # axsTC.legend(loc='best', fontsize=80, frameon=False)
    # # axsTC.set_xlabel(r'time ($\theta$)', fontsize=80)
    # # axsTC.set_ylabel(r'travel time', fontsize=80)
    # # plt.setp(axsTC.get_xticklabels(), fontsize=80)


    # #-----------------
    # # Alpha and FlowDiff per iteration
    # #-----------------
    # alphaIter = data['alphaIter']
    # absDiffBwFlowsIter = data['absDiffBwFlowsIter']
    # relDiffBwFlowsIter = data['relDiffBwFlowsIter']
    # qopiIter = data['qopiIter']
    # # qopiMeanIter = data['qopiMeanIter']
    # qopiFlowIter = data['qopiFlowIter']
    # # a,b = [round(float(c),2) for c in alphaIter],[round(float(c),2) for c in diffBwFlowsIter]
    # # print(a,b)
    # x = [x+1 for x in range(len(alphaIter))]

    # k += 1
    # axs[k].plot(x,alphaIter,label=r'$\alpha$', color=colors[0], linestyle=linestyles[0])
    # axs2 = axs[k].twinx()
    # axs2.plot(x,absDiffBwFlowsIter,label=r'$\Delta$ f', color=colors[1], linestyle=linestyles[1])
    # axs2.plot(x,relDiffBwFlowsIter,label=r'($\Delta$ f / f)', color=colors[2], linestyle=linestyles[1])
    # axs2.plot([1,len(alphaIter)], [float(precision), float(precision)],label=r'$\epsilon$',\
            # color=colors[3], linestyle=linestyles[2])
    # axs2.legend(loc=locs[3], fontsize='x-large')
    # # axs[k].set_xlabel('iteration', fontsize='xx-large')
    # # axs[k].set_title(r'$\alpha$ and $\Delta f^{k}$', fontsize='xx-large')
    # # plt.show()
   # # TODO: Avoid this hardcoding
    # # axs[2].legend(loc=locs[2])
    # axs[k].legend(loc=locs[1], fontsize='x-large')

    # k += 1
    # # axs[k].plot(x,qopiIter,label='QoPI', color=colors[3], linestyle=linestyles[1])
    # # axs[k].plot(x,qopiMeanIter,label='QoPIMean', color=colors[4], linestyle=linestyles[1])
    # axs[k].plot(x,qopiFlowIter,label='QoPI (per unit flow)', color=colors[4], linestyle=linestyles[1])
    # axs[k].set_xlabel('iteration', fontsize='xx-large')
    # axs[k].legend(fontsize='x-large')

    # # Set label and xtick sizes for axes
    # for i in range(len(axs)):
        # # axs[i].legend(fontsize='x-large')
        # plt.setp(axs[i].get_xticklabels(), fontsize='x-large')
        # plt.setp(axs[i].get_yticklabels(), fontsize='x-large')

    # plt.ylim(bottom=0)

    # # mng = plt.get_current_fig_manager()
    # # mng.full_screen_toggle()
    # figs = []
    # for i in plt.get_fignums():
        # # print(plt.gcf, plt.fignum_exists(i), i)
        # mng = plt.figure(i).canvas.manager
        # mng.full_screen_toggle()
        # # Save reference to the figures, else they are deleted after plt.show()
        # figs.append(plt.figure(i))

## Multicommodity Plots
#-----------------
# QoPI and FlowDiff per iteration
#-----------------
# figQ, axsQ = plt.subplots(1)
figD, axsD = plt.subplots(1)
axsQ = axsD.twinx()

absDiffBwFlowsIter = data['absDiffBwFlowsIter']
qopiFlowIter = data['qopiFlowIter']
alphaIter = data['alphaIter']
relDiffBwFlowsIter = data['relDiffBwFlowsIter']
qopiIter = data['qopiIter']

# Adjust inf
# if qopiFlowIter[-1] == math.inf:
    # qopiFlowIter[-1] = qopiFlowIter[-2]
    # print(qopiFlowIter)
# else:
    # print(qopiFlowIter[-1])
    # exit(0)

prec = float(precision)
x = [x+1 for x in range(len(alphaIter))]

l1 = axsD.plot(x,absDiffBwFlowsIter, color=colors[1], linewidth=10, linestyle=linestyles[1], label=r'$\Delta$h')
l2 = axsD.plot([1,len(alphaIter)], [float(precision), float(precision)],label=r'$\epsilon$',\
            color='red', linewidth=10, linestyle=linestyles[2])
# axsD.axhline(prec, xmin=0.05, xmax=0.95, label=r'$\epsilon$', color='red', linewidth=10, linestyle=linestyles[2])
l3 = axsQ.plot(x,qopiFlowIter, color=colors[4], linewidth=10, linestyle=linestyles[1], label='QoPI')
trans = transforms.blended_transform_factory(
            axsD.get_yticklabels()[0].get_transform(), axsD.transData)
# axsD.text(0,float(precision), "{:.2f}".format(prec), color="red", transform=trans, 
                # ha="right", va="center", fontsize=80)
# axsD.text(1, float(precision), r'$\epsilon$', color="red", transform=trans, 
                # ha="right", va="center", fontsize=80)
# axsD.legend(loc='best', fontsize=80, frameon=False)
# axsQ.legend(loc='center right', fontsize=80, frameon=False)
# axsD.annotate(r'\epsilon', xy=[0,0.01], xytext=(-1,0.01),textcoords='test')

axsD.set_xlabel(r'iteration', fontsize=80)
axsD.set_ylabel(r'$\Delta$h', fontsize=80)

# axsQ.legend(loc='best', fontsize=80, frameon=False)
# axsQ.set_xlabel(r'iteration', fontsize=80)
axsQ.set_ylabel(r'QoPI', fontsize=80)

axsD.set_xscale('log', basex=2)
axsD.xaxis.set_tick_params(pad=20)
axsD.set_xscale('log', basex=2)
axsD.xaxis.set_tick_params(pad=20)

plt.setp(axsD.get_xticklabels(), fontsize=80)
plt.setp(axsD.get_yticklabels(), fontsize=80)
plt.setp(axsQ.get_xticklabels(), fontsize=80)
plt.setp(axsQ.get_yticklabels(), fontsize=80)

# plt.legend(loc='best', fontsize=80, frameon=False)
# For a common legend
lns = l3 + l1 + l2
labs = [l.get_label() for l in lns]
axsD.legend(lns, labs, loc='best', fontsize=80, frameon=False)


# ENERGY PROFILES
bmax = 30
figB, axsB = plt.subplots(1)
yBsum1 = []
for c,p in enumerate(f[()].fPlus):
    print('Comm.',c)
    u = sum([f[()].fPlus[c][p1].integrate(0,1) for p1 in f[()].fPlus[c]])
    # bmax = 1 + max([2*p.getNetEnergyConsump() for p in f[()].fPlus[c]])
    # bmax =13
    # bmin = min([p.getNetEnergyConsump() for p in f[()].fPlus[c]])
    # for p in f[()].fPlus[c]:
      # # print(p, p.getNetEnergyConsump())
    # print('bmin ',bmin)
    yBsum = []
    # yBsum1 = []
    tt = data['travelTime']
    xB1 = [float(timeStep)/2 + x*float(timeStep) for x in range(int((len(tt[0][0])-0)))]

    for i,p1 in enumerate(f[()].fPlus[c]):
        x,y = f[()].fPlus[c][p1].getXandY(0,f[()].getEndOfInflow(c))
        # yB = [p1.getNetEnergyConsump()*v/u for v in y]
        y1 = [f[()].fPlus[c][p1].getValueAt(i) for i in xB1]
        yB1 = [p1.getNetEnergyConsump()*v/u for v in y1]
        print('x', x)
        print('y', y)
        print('xB1', xB1)
        print('yB1', yB1)
        # TODO: set a compact logic for the following line
        if (len(y) > 2) or (len(y) > 1 and y[0]>0):
            xB = x
            # [lambda: value_false, lambda: value_true][<test>]()
            # yBsum = [lambda:yB, lambda:[yBsum[i]+yB[i] for i,_ in enumerate(yB)]][len(yBsum)>0]()
            yBsum1 = [lambda:yB1, lambda:[yBsum1[i]+yB1[i] for i,_ in enumerate(yB1)]][len(yBsum1)>0]()
            print('i', i, p1.getNetEnergyConsump())
            # print(yB)
            # axsB.plot(x,yB,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1], linewidth=10)
        # print('i', i)
        # print('y', len(y), y, p1.getNetEnergyConsump())
        # print('yB', len(yB), yB)
        print('yBsum1', len(yBsum1), yBsum1)
    print('yBsum1', len(yBsum1), yBsum1)

      # a,b = [int(c) for c in x],[int(c) for c in y]
      # print("i: ", i,a,b)
      # axsB.plot(x,yB,label='Total', color=colors[i], linestyle=linestyles[1],
              # linewidth=10)
print('yBsum1', yBsum1)
axsB.plot(xB1,yBsum1,label=r'no r/c', color='cyan', linestyle=linestyles[1],
      linewidth=10)
# axsB.legend(loc='best', fontsize=80, frameon=False)
axsB.set_xlabel(r'time ($\theta$)', fontsize=80)

# Temporary: uncomment if y-ticks and y-labels are not needed
# axsB.set_ylabel(r'Battery Consump. / Unit Flow', fontsize=80)

# axsB.set_ylim([0, bmax])
# axsB.set_ylim(bottom=float(energyBudget))

plt.setp(axsB.get_xticklabels(), fontsize=80)
plt.setp(axsB.get_yticklabels(), fontsize=80)

### ADDITIONAL ENERGY PROFILES
# Reading another data file
if len(sys.argv) > 2:
    for j in range(len(sys.argv)-2):
        print('j',j,len(sys.argv))
        data1 = np.load(sys.argv[2+j], allow_pickle=True);
        f1 = data1['f']
        fname1 = os.path.splitext(os.path.split(sys.argv[2+j])[1])[0]
        print(re.split('[_]', fname1))
        [insName1,timeHorizon1,maxIter1,timeLimit1,precision1,alpha1,timeStep1,priceToTime,numThreads,alphaStr1] = re.split('[_]', fname1)

        yBsum1 = []
        for c,p in enumerate(f[()].fPlus):
            u = sum([f[()].fPlus[c][p1].integrate(0,1) for p1 in f[()].fPlus[c]])
            yBsum = []
            tt = data1['travelTime']
            xB1 = [float(timeStep)/2 + x*float(timeStep) for x in range(int((len(tt[0][0])-0)))]
            for i,p1 in enumerate(f1[()].fPlus[c]):
                x,y = f1[()].fPlus[c][p1].getXandY(0,f[()].getEndOfInflow(c))
                y1 = [f1[()].fPlus[c][p1].getValueAt(i) for i in xB1]
                # yB = [p1.getNetEnergyConsump()*v/u for v in y]
                yB1 = [p1.getNetEnergyConsump()*v/u for v in y1]
                print('x', x)
                print('y', y)
                print('xB1', xB1)
                print('yB1', yB1)
                if (len(y) > 2) or (len(y) > 1 and y[0]>0):
                  xB = x
                  # [lambda: value_false, lambda: value_true][<test>]()
                  yBsum1 = [lambda:yB1, lambda:[yBsum1[i]+yB1[i] for i,_ in
                      enumerate(yB1)]][len(yBsum1)>0]()
                  print('i', i, p1.getNetEnergyConsump())
                  # print(yB)
                  # axsB.plot(x,yB,label='w%d'%(i+1), color=colors[i+1], linestyle=linestyles[1], linewidth=10)
        print('yBsum1', yBsum1)
        axsB.plot(xB1,yBsum1,label=r'$\widetilde{\lambda}_i = %d$'%int(priceToTime), color=colors[j], linestyle=linestyles[1],
              linewidth=10)
        # axsB.plot(xB1,yBsum1,label=r'with recharging', color=colors[j], linestyle=linestyles[1],
              # linewidth=10)
        # axsB.plot(xB,[float(energyBudget) for i in xB], label=r'$b^{max}$', color=colors[-2],
                # linestyle='dotted', linewidth=10)
axsB.legend(loc='best', fontsize=80, frameon=False, ncol=3)
# axsB.set_xlabel(r'time ($\theta$)', fontsize=80)

axsB.set_ylabel(r'Battery Consump. / Unit Flow', fontsize=80)
# axsB.set_ylim([0, bmax])
# axsB.set_ylim(bottom=float(15),top=float(25))
axsB.set_ylim([15, 25])
# axsB.set_ylim(bottom=float(energyBudget))
# axsB.set_ylim([float(energyBudget), bmax])

# plt.setp(axsB.get_xticklabels(), fontsize=80)
# plt.setp(axsB.get_yticklabels(), fontsize=80)

###############
# TRAVEL TIMES
# figTC, axsTC = plt.subplots(1)
# for c,p in enumerate(f[()].fPlus):
    # print('Comm.',c)
    # tt = data['travelTime']
    # # print(tt)
    # # print(tt[0][0], len(tt), len(tt[0]), len(tt[0][0]))
    # # print(tt[c], len(tt[c]), len(tt[c][0]))

    # x = [float(timeStep)/2 + x*float(timeStep) for x in\
        # range(int((len(tt[0][0])-0)))]
    # ttmax = np.amax(tt[c]) + 1
    # tmin = []
    # for p in range(len(tt[c])):
        # y = tt[c][p]
        # tmin = [lambda:y, lambda:[min(tmin[i],y[i]) for i,_ in enumerate(y)]][p>0]()
    # print('\ntmin', tmin)
    # print('x', x)
    # # print('len tmin', len(tmin), len(x))
    # axsTC.plot(x,tmin,label='comm%d'%c, color=colors[c], linestyle=linestyles[1],
            # linewidth=10)
    # axsTC.legend(loc='best', fontsize=80, frameon=False)
    # axsTC.set_xlabel(r'time ($\theta$)', fontsize=80)
    # axsTC.set_ylabel(r'Min. Travel Time', fontsize=80)
    # axsTC.set_ylim([400, 2500])
    # plt.setp(axsTC.get_yticklabels(), fontsize=80)
    # plt.setp(axsTC.get_xticklabels(), fontsize=80)


###############

figs = []
for i in plt.get_fignums():
    mng = plt.figure(i).canvas.manager
    mng.full_screen_toggle()
    # Save reference to the figures, else they are deleted after plt.show()
    figs.append(plt.figure(i))

print('-------')
print('Summary')
print('-------')
print("Termination message: ", data['stopStr'])
print("\nAttained DiffBwFlows (abs.): %.5f"%absDiffBwFlowsIter[-2])
print("Attained DiffBwFlows (rel.): %.5f"%relDiffBwFlowsIter[-2])
print("\nAttained QoPI (abs.): %.5f"%qopiIter[-2])
# print("Attained QoPI (mean): %.4f"%qopiMeanIter[-2])
print("Attained QoPI (per unit flow): %.5f"%qopiFlowIter[-2])
print("\nIterations : ", len(alphaIter))
print("Elapsed wall time: ", runTime)


# Save figures
dirname = os.path.expanduser('./figures')
# fname1 = fname + '_comm%d'%c
# print(fname)
plt.show()
for i,fig in enumerate(figs):
    figname = os.path.join(dirname, fname)
    if i == 0:
        figname += '_combQoPIFlowDiff'
    elif i == 1:
        figname += '_combEnerProfs'
    elif i == 2:
        figname += '_commTravTimes'
    # elif i == 3:
        # figname += '_travTimes'
    # elif i == 4:
        # figname += '_qopiWalks'
    figname += '.png'
    print("\noutput saved to file: %s"%figname)
    if i == -1:
        fig.savefig(figname, format='png', dpi=fig.dpi, bbox_inches='tight')
plt.close()
