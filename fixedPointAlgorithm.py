from typing import List, Dict, Tuple

from networkloading import *
from dynamic_dijkstra import dynamic_dijkstra

# For the root finding problem
from scipy import optimize
import sys, time
import numpy as np

def findShortestSTpath(s: Node, t: Node, flow: PartialFlow, time: number) -> Path:
    (arrivalTimes, realizedTimes) = dynamic_dijkstra(time, s, t, flow)
    # for i in enumerate(arrivalTimes):
        # print("times ", i, i[0], i[1])
    # for i in enumerate(realizedTimes):
        # print("times ", i, i[0], i[1])
    p = Path([], t)
    while p.getStart() != s:
        v = p.getStart()
        for e in v.incoming_edges:
            if e in realizedTimes and arrivalTimes[e.node_from] + realizedTimes[e] == arrivalTimes[v]:
                p.add_edge_at_start(e)
                break

    # print("shortest ST path ", printPathInNetwork(p, flow.network))
    return p


def findShortestFeasibleSTpath(time: number, s: Node, t: Node, flow:
        PartialFlow, budget: number) -> Path:
    (arrivalTimes, realizedTimes) = dynamicFeasDijkstra(time, s, t, flow, budget)
    for i in enumerate(arrivalTimes):
        print("times ", i, i[0], i[1])
    for i in enumerate(realizedTimes):
        print("times ", i, i[0], i[1])
    p = Path([], t)
    while p.getStart() != s:
        v = p.getStart()
        for e in v.incoming_edges:
            if e in realizedTimes and arrivalTimes[e.node_from] + realizedTimes[e] == arrivalTimes[v]:
                p.add_edge_at_start(e)
                break

    print("shortest ST path ", printPathInNetwork(p, flow.network))
    return p


# def setInitialPathFlows(G: Network, pathList : List[Path],\
        # commodities : List[Tuple[Node, Node, PWConst]],\
        # zeroflow: PartialFlow, pathInflows: PartialFlowPathBased) -> PartialFlowPathBased:
    # print("To be implemented! Passing hardcoded path inflows.")
    # genPaths = False
    # if not pathList:
        # genPaths = True

    # for i,(s,t,u) in enumerate(commodities):
        # # TODO: get rid of this temporary hack asap
        # # Get pathlist if empty
        # if genPaths:
            # # pathList = getEVExamplePaths(G, s, t)
            # # pathList = getLeonsPaths(G, s, t)
            # pathList = getNguyenPaths(G, s, t)

        # # Get flowlist
        # # flowlist = [u,PWConst([0,50],[0],0),PWConst([0,50],[0],0)]
        # # flowlist = [PWConst([0,50],[0],0)]*(len(pathList)-1)
        # flowlist = [PWConst([0,u.segmentBorders[-1]],[0],0)]*(len(pathList)-1)
        # flowlist.insert(0,u)
        # # print("len ", len(pathlist), len(flowlist))
        # # exit(0)

        # # pathInflows.setPaths(commodityId, [p1,p2,p3], [u,PWConst([0,50],[0],0),PWConst([0,50],[0],0)])
        # print("Setting paths up for s-t commodity: ", s, "-", t)
        # pathInflows.setPaths(i, pathList, flowlist)
        # # print(pathlist, flowlist)
        # # print("Setting up path ", p2)
        # # pathInflows.setPaths(commodityId, p2, 0)
        # # print("Setting up path ", p3)
        # # pathInflows.setPaths(commodityId, p3, 0)
    # return pathInflows


def fixedPointUpdate(currentFlow: PartialFlow, oldPathInflows: PartialFlowPathBased, timeHorizon:
        number, alpha: float, timestepSize, commodities, verbose: bool) -> PartialFlowPathBased:
    # currentFlow = networkLoading(oldPathInflows)

    newPathInflows = PartialFlowPathBased(oldPathInflows.network, oldPathInflows.getNoOfCommodities())

    # record the difference of derived times and shortest path times
    # timeDiff = [[0 for i in range(int(timeHorizon/timestepSize))] for j in
            # range(oldPathInflows.getNoOfCommodities())]
    # print("timeDiff ", len(timeDiff),
            # range(oldPathInflows.getNoOfCommodities()),range(int(timeHorizon/timestepSize)),timeDiff)
    # for i in range(oldPathInflows.getNoOfCommodities()):
    for i,comd in enumerate(commodities):
        flowValue = [None]*len(oldPathInflows.fPlus[i])
        travelTime = [None]*len(oldPathInflows.fPlus[i])
        if False: print("Considering commodity ", i)
        newPathInflows.setPaths(i,[P for P in oldPathInflows.fPlus[i]],[PWConst([zero],[],zero) for P in oldPathInflows.fPlus[i]])
        s = newPathInflows.sources[i]
        t = newPathInflows.sinks[i]
        theta = zero
        meanIter = 0
        # We subdivide the time into intervals of length timestepSize
        k = -1
        # alphaList = []
        # alpha = alphaComm[i]
        while theta < oldPathInflows.getEndOfInflow(i):
            k += 1
            # For each subinterval i we determine the dual variable v_i
            # (and assume that it will stay constant for the whole interval)
            # if verbose: print("timeinterval [", theta, ",", theta+timestepSize,"]")

	    # Set up the update problem for each subinterval
            # print("\nSetting up the update problem for subinterval [", theta,
                    # ",", theta+timestepSize,"]\n")
            # maxTravelTime = 0
            # Get path travel times for this subinterval
            for j,P in enumerate(oldPathInflows.fPlus[i]):
                 fP = oldPathInflows.fPlus[i][P]
                 # converting to float (optimize.root does not work with fractions)
                 travelTime[j] = float(currentFlow.pathArrivalTime(P,\
                     theta + timestepSize/2) - (theta + timestepSize/2))
                 flowValue[j] = float(fP.getValueAt(theta))
                 # maxTravelTime = max(maxTravelTime, travelTime[j])
                 # print("Path: ",printPathInNetwork(P,currentFlow.network),
                         # "flowValue: ", round(float(flowValue[j]),2), "travelTime: ",\
                         # travelTime[j], "at theta =",\
                         # round(float(theta + timestepSize/2),2), "fp: ", fP)

            # Compare with the shortest path travel time
            # shortestPath = findShortestSTpath(s,t,currentFlow,theta + timestepSize/2)
            # shortestTravelTime = currentFlow.pathArrivalTime(shortestPath,\
                    # theta+timestepSize/2)- (theta + timestepSize/2)
            # print("shortest path ", shortestPath.getFreeFlowTravelTime(), round(float(theta +\
                             # timestepSize/2),2), printPathInNetwork(shortestPath,\
                             # currentFlow.network), round(float(shortestTravelTime)))
            # timeDiff[i][k] = maxTravelTime - shortestTravelTime
            # print("check ", i,k,len(timeDiff),timeDiff[0][0])
            # if timeDiff[i][k] < 0:
                # print("maxTravelTime %.2f less than shortestTravelTime %.2f!"\
                        # % (float(maxTravelTime),float(shortestTravelTime)))
                # exit(0)

            # Find integral value, ubar, of (piecewise constant) function u in this
            # subinterval
            ubar = comd[2].integrate(theta, theta + timestepSize)

            # For adjusting alpha
            # uval = ubar/timestepSize
            # alpha = uval/min(travelTime)
            # alpha = 0.5*uval*(1/min(travelTime) + 1/max(travelTime))
            # alpha = 0.5*uval*(1/max(travelTime))
            # alpha = uval/max(travelTime)
            # adjAlpha = [uval/min(travelTime) for j,_ in enumerate(flowValue)]
            # alphaList.append(adjAlpha[0])
            # print('adjAlpha: ', theta + timestepSize/2, uval, [round(i, 4) for
                # i in travelTime], [round(i,4) for i in adjAlpha])
            # adjAlpha = [alpha*flowValue[j]/(2*travelTime[j]) for j,_ in enumerate(flowValue)]
            # adjAlpha = [flowValue[j]/(2*travelTime[j]) for j,_ in enumerate(flowValue)]

            # adjmin = min([i for i in adjAlpha if i > 0])
            # adjmax = max([i for i in adjAlpha if i > 0])
            # adjAlpha = [adjmin if j == 0 else j for j in adjAlpha]
            # adjAlpha = [adjmax if j == 0 else j for j in adjAlpha]

            # TODO: Find a good starting point
            # A trivial guess: assume all terms to be positive and solve for the dual variable
            x0 = ((-sum(flowValue) + alpha*sum(travelTime))*timestepSize +
                    ubar)/(len(flowValue)*timestepSize)
            # print("x0 ", round(x0,2))
            # optimize.show_options(solver='root', method='broyden1', disp=True)
            # TODO: Find a way to run optimize.root quietly
            bracketLeft = 0
            bracketRight = abs(max(list(map(float.__sub__, list(map(lambda x: alpha*x,
                travelTime)), flowValue)))) + ubar + 1

            # Default (brentq) method using bracket
            # sol = optimize.root_scalar(dualVarRootFunc, (alpha, flowValue, travelTime,
                # timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight])

            # Newton's method using separate routines for value and derivative
            # sol = optimize.root_scalar(dualVarRootFunc, (alpha, flowValue, travelTime,
                # timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight],
                # fprime=dualVarRootFuncGrad, method='newton')

            # Newton's method using a routine that return value and derivative
            # sol = optimize.root_scalar(dualVarRootFuncComb, (adjAlpha, flowValue, travelTime,
            sol = optimize.root_scalar(dualVarRootFuncComb, (alpha, flowValue, travelTime,
                timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight],
                fprime=True, method='newton')

            if not sol.converged:
                print("The optimize.root_scalar() method has failed to converge due to the following reason:")
                print("\"", sol.flag, "\"")
                alpha = alpha/2 + uval/(2*max(travelTime))
                sol = optimize.root_scalar(dualVarRootFuncComb, (alpha, flowValue, travelTime,
                    timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight],
                    fprime=True, method='newton')
                if not sol.converged:
                    print("Adjusted alpha! The optimize.root_scalar() method has still failed to converge because:")
                    print("\"", sol.flag, "\"")
                    exit(0)
                else:
                    meanIter += sol.iterations
            else:
                meanIter += sol.iterations
                # print(sol)
            # print("currentFlow ", currentFlow)
            for j,P in enumerate(oldPathInflows.fPlus[i]):
                newFlowVal = max(flowValue[j] - alpha*travelTime[j] + sol.root, 0)
                # newFlowVal = max(flowValue[j] - adjAlpha[j]*travelTime[j] + sol.root, 0)
                # print("newFlowVal ", newFlowVal)
                newPathInflows.fPlus[i][P].addSegment(makeNumber(theta+timestepSize), makeNumber(newFlowVal))
            # print("newPathInflows: ", newPathInflows)
            theta = theta + timestepSize
        tmpVar = max(timestepSize,1/timestepSize)
        if False: print("Mean # of root.scalar() iterations ",\
                float(round(meanIter/(tmpVar*oldPathInflows.getEndOfInflow(i)),2)),\
                " for ", tmpVar*oldPathInflows.getEndOfInflow(i), " subintervals")
    # for id, e in enumerate(currentFlow.network.edges):
        # print("queue at edge %d: "%id, e, currentFlow.queues[e])
    # print("timeDiff ", timeDiff)
    # print("newPathInflows: ", newPathInflows)
    # return newPathInflows, sum(alphaList)/len(alphaList)
    return newPathInflows, alpha

def dualVarRootFunc(x, alpha, flowValue, travelTime, timestepSize, ubar):
    # print("printing args ", round(x,2),alpha,flowValue,travelTime,timestepSize,ubar)
    termSum = 0
    for j,fv in enumerate(flowValue):
        # print("print terms for j", j, " : ", flowValue[j],alpha,travelTime[j],x,timestepSize)
        termSum += max(flowValue[j] - alpha*travelTime[j] + x, 0)*timestepSize
        # print("termSum in loop ", termSum)
    # print("result ", termSum, ubar, termSum - ubar)
    return float(termSum - ubar)


def dualVarRootFuncGrad(x, alpha, flowValue, travelTime, timestepSize, ubar):
    termSum = 0
    for j,fv in enumerate(flowValue):
        if (flowValue[j] - alpha*travelTime[j] + x) > 0:
            termSum += timestepSize
    return float(termSum)


def dualVarRootFuncComb(x, alpha, flowValue, travelTime, timestepSize, ubar):
    # print("printing args ", round(x,2),alpha,flowValue,travelTime,timestepSize,ubar)
    termSum = 0
    gradTermSum = 0
    for j,fv in enumerate(flowValue):
        tmp = flowValue[j] - alpha*travelTime[j] + x
        # print('tmp %.2f %.2f %.2f %.2f %.2f'%(flowValue[j], alpha, travelTime[j], x, tmp))
        # tmp = flowValue[j] - alpha[j]*travelTime[j] + x
        if tmp > 0:
            termSum += tmp*timestepSize
            gradTermSum += timestepSize
    return float(termSum - ubar), float(gradTermSum)


def differenceBetweenPathInflows(oldPathInflows : PartialFlowPathBased, newPathInflows : PartialFlowPathBased) -> number:
    assert (oldPathInflows.getNoOfCommodities() == newPathInflows.getNoOfCommodities())
    #TODO: Also check if the time horizon for both the pathInflows is same or not

    difference = zero

    for i in range(oldPathInflows.getNoOfCommodities()):
        for path in oldPathInflows.fPlus[i]:
            if path in newPathInflows.fPlus[i]:
                difference += (oldPathInflows.fPlus[i][path] + newPathInflows.fPlus[i][path].smul(-1)).norm()
            else:
                difference += oldPathInflows.fPlus[i][path].norm()
        for path in newPathInflows.fPlus[i]:
            if path not in oldPathInflows.fPlus[i]:
                difference += newPathInflows.fPlus[i][path].norm()

    return difference

# Find the sum of norm (integration) of path inflow fPlus functions
def sumNormOfPathInflows(pathInflows : PartialFlowPathBased) -> number:
    sumNorm = zero
    for i in range(pathInflows.getNoOfCommodities()):
        for path in pathInflows.fPlus[i]:
            sumNorm += (pathInflows.fPlus[i][path]).norm()
    # print("sumNorm: ", sumNorm)
    # TODO: This should be equal to the integration of u (if required, put an assert)
    return sumNorm

# Function arguments: (network, precision, List[source node, sink node, ?], time
# horizon, maximum allowed number of iterations, verbosity on/off)
# TODO: warm-start using an available path flow?
def fixedPointAlgo(N : Network, pathList : List[Path], precision : float, commodities :
        List[Tuple[Node, Node, PWConst]], timeHorizon:
        number=infinity, maxSteps: int = None, timeLimit: int = infinity, timeStep: int = None,
        alpha : float = None, verbose : bool = False) -> PartialFlowPathBased:
    tStartAbs = time.time()
    step = 0

    ## Init:
    # Create zero-flow (PP: why?)
    pathInflows = PartialFlowPathBased(N,0)
    # TODO: Conform with LG if this can be removed
    # zeroflow = networkLoading(pathInflows)

    pathInflows = PartialFlowPathBased(N, len(commodities))
    # Initial flow: For every commodity, select the shortest s-t path and send
    # all flow along this path (and 0 flow along all other paths)
    # alpha_0 = []
    for i,(s,t,u) in enumerate(commodities):
        flowlist = [PWConst([0,u.segmentBorders[-1]],[0],0)]*(len(pathList[i])-1)
        flowlist.insert(0,u)
        # print("set ", u, flowlist, pathList[i], pathInflows)
        pathInflows.setPaths(i, pathList[i], flowlist)
        # alpha_0.append(u.getValueAt(0)/min([p.getFreeFlowTravelTime() for p in
            # pathList[i]]))
        # print("u ", u)
        # setInitialPathFlows(i, N, s, t, u, zeroflow, pathInflows)

    if False: print("Starting with flow: \n", pathInflows)

    oldAbsDiffBwFlows = infinity
    oldRelDiffBwFlows = infinity
    gamma = makeNumber(1)
    alphaIter = []
    absDiffBwFlowsIter = []
    relDiffBwFlowsIter = []
    travelTime = []
    qopiIter = []  # qualityOfPathInflows
    qopiMeanIter = []  # mean of qopi
    shouldStop = not (maxSteps is None or step < maxSteps)


    # alphaStr = ''
    # alphaStr = 'uByTmin'
    # alphaStr = 'uBy2Tmin'
    # alphaStr = 'meanUByTminTmax'
    # alphaStr = 'uByTmax'
    # alphaStr = r'($\gamma$)'
    # alphaStr = r'($\gamma\alpha$)'
    alphaStr = r'expoSmooth($\gamma$)'
    # alphaStr = r'expoSmooth($\gamma/2$)'
    # alphaStr = r'relExpoSmooth($\gamma/2$)'
    # alphaStr = r'min2ExpoSmooth($\gamma/2$)'

    tStart = time.time()
    iterFlow = networkLoading(pathInflows)
    print("\nTime taken in networkLoading(): ", round(time.time()-tStart,4))

    # flowVal = commodities[0][2].segmentValues[0]
    ## Iteration:
    while not shouldStop:
        if verbose: print("STARTING ITERATION #", step)
        # newpathInflows = networkLoading(pathInflows,timeHorizon)
        # print("newpathInflows ", newpathInflows)
        tStart = time.time()
        newpathInflows, alpha = fixedPointUpdate(iterFlow, pathInflows, timeHorizon, alpha,
                timeStep, commodities, verbose)
        print("\nTime taken in fixedPointUpdate(): ", round(time.time()-tStart,4))

        newAbsDiffBwFlows = differenceBetweenPathInflows(pathInflows,newpathInflows)
        newRelDiffBwFlows = newAbsDiffBwFlows/sumNormOfPathInflows(pathInflows)

        # Check Stopping Conditions
        if newAbsDiffBwFlows < precision:
            shouldStop = True
            stopStr = "Attained required (absolute) precision!"
        # elif newRelDiffBwFlows < precision/10:
            # shouldStop = True
            # stopStr = "Attained required (relative) precision!"
        elif not (maxSteps is None or step < maxSteps):
            shouldStop = True
            stopStr = "Maximum number of steps reached!"

        elif (time.time() - tStartAbs > timeLimit):
            shouldStop = True
            stopStr = "Maximum time limit reached!"

        qopi = infinity
        qopiMean = infinity
        if not shouldStop:
            # Update Alpha
            if newAbsDiffBwFlows == 0:
                gamma = 0
            else:
                if step > 0: gamma = 1 - abs(newAbsDiffBwFlows - oldAbsDiffBwFlows)/(newAbsDiffBwFlows +
                        oldAbsDiffBwFlows)

            # Alpha Update Rule
            # oldalpha = alpha
            # alpha = gamma # equal to factor
            # alpha = gamma*alpha # multiplied by factor
            alpha = (gamma)*(gamma*alpha) + (1-gamma)*alpha # expo smooth using gamma
            # alpha = (0.5*gamma)*(0.5*gamma*alpha) + (1-0.5*gamma)*alpha # expo smooth using gamma/2
            # alpha = max(0.2, (0.5*gamma)*(0.5*gamma*alpha) + (1-0.5*gamma)*alpha) # expo smooth using gamma/2
            # if step > 1 and oldalpha == alpha:
                # print('Changing alpha')
                # alpha = alpha*(0.5) #step/maxSteps

            # Measure quality of the path inflows
            tStart = time.time()
            iterFlow = networkLoading(newpathInflows)
            tEnd = time.time()
            print("\nTime taken in networkLoading(): ", round(tEnd-tStart,4))

            qopi = 0
            qopiMean = 0
            for i,comd in enumerate(commodities):
                tminmin = infinity
                if False: print('comm ', i)
                fP = newpathInflows.fPlus[i]
                theta = zero
                while theta < newpathInflows.getEndOfInflow(i):
                    tt = np.empty(len(newpathInflows.fPlus[i]))
                    for j,P in enumerate(newpathInflows.fPlus[i]):
                        tt[j] = iterFlow.pathArrivalTime(P,theta + timeStep/2) - (theta + timeStep/2)
                    tmin = min(tt)
                    tminmin = min(tminmin, tmin)
                    # qopiTheta = []
                    fval = []
                    for j,P in enumerate(newpathInflows.fPlus[i]):
                        fval.append(fP[P].getValueAt(theta + timeStep/2))
                        # qopiTheta.append((tt[j] - tmin)*fval[j])
                        qopi += (tt[j] - tmin)*fP[P].getValueAt(theta + timeStep/2)
                        # if False: print(j, tt[j], tmin, fP[P].getValueAt(theta + timeStep/2), qopi)
                    # print('theta ', theta + timeStep/2, [round(i,4) for i in tt],
                        # [round(i,4) for i in fval], [round(i,4) for i in qopiTheta])
                    # print('\n')
                    theta = theta + timeStep
                qopiMean += qopi/(pathInflows.getEndOfInflow(i)/timeStep)
            if verbose: print("Norm of change in flow (abs.) ", round(float(newAbsDiffBwFlows),4),\
                    " previous change ", round(float(oldAbsDiffBwFlows),4), " alpha ",\
                    round(float(alpha),4), ' qopi ', round(qopi,4), ' qopiMean ',\
                    round(qopiMean,4))
            if verbose: print("Norm of change in flow (rel.) ", round(float(newRelDiffBwFlows),4),\
                    " previous change ", round(float(oldRelDiffBwFlows),4))

            # alpha = flowVal/tminmin
            # print(flowVal, tminmin, alpha)

            # if verbose: print("Current flow is\n", newpathInflows)
            # update iteration variables
            pathInflows = newpathInflows
            oldAbsDiffBwFlows = newAbsDiffBwFlows
            oldRelDiffBwFlows = newRelDiffBwFlows

        qopiIter.append(qopi)
        qopiMeanIter.append(qopiMean)
        # alphaIter.append(alphaList)
        alphaIter.append(alpha)
        absDiffBwFlowsIter.append(newAbsDiffBwFlows)
        relDiffBwFlowsIter.append(newRelDiffBwFlows)

        step += 1
        # print('pathInflows: ', pathInflows)
        # print("\nEND OF ITERATION ", step,"\n")

    print(stopStr)
    # Find path travel times for the final flow
    finalFlow = networkLoading(pathInflows, verbose=False)
    for i,comd in enumerate(commodities):
        ttravelTime = np.empty([len(pathInflows.fPlus[i]),\
                math.ceil(pathInflows.getEndOfInflow(i)/timeStep)])
        theta = zero
        k = -1
        while theta < pathInflows.getEndOfInflow(i):
            k += 1
            for j,P in enumerate(pathInflows.fPlus[i]):
                 # fP = pathInflows.fPlus[i][P]
                 ttravelTime[j][k] = finalFlow.pathArrivalTime(P,\
                     theta + timeStep/2) - (theta + timeStep/2)
            theta = theta + timeStep
        # print("ttravelTime", np.shape(ttravelTime), ttravelTime)
        travelTime.append(ttravelTime)
    # print("travelTime", travelTime)
    # print("qopiIter ", qopiIter)
    return pathInflows, alphaIter, absDiffBwFlowsIter, relDiffBwFlowsIter,\
            travelTime, stopStr, alphaStr, qopiIter, qopiMeanIter

