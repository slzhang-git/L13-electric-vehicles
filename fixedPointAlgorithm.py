from typing import List, Dict, Tuple

from networkloading import *
from dynamic_dijkstra import dynamic_dijkstra

# For the root finding problem
from scipy import optimize

def findShortestSTpath(s: Node, t: Node, flow: PartialFlow, time: ExtendedRational) -> Path:
    (arrivalTimes, realizedTimes) = dynamic_dijkstra(time, s, t, flow)
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


def findShortestFeasibleSTpath(time: ExtendedRational, s: Node, t: Node, flow:
        PartialFlow, budget: ExtendedRational) -> Path:
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


def getEVExamplePaths(G: Network, s: Node, t: Node) -> List[Path]:
    # Paths for EVExample3 network
    p1 = Path([G.edges[0], G.edges[2], G.edges[4]])
    p2 = Path([G.edges[1], G.edges[2], G.edges[3]])
    p3 = Path([G.edges[1], G.edges[2], G.edges[4]])
    # printPathInNetwork(p2,G)
    # printPathInNetwork(p3,G)
    # exit(0)
    return [p1, p2, p3]


def getLeonsPaths(G: Network, s: Node, t: Node) -> List[Path]:
    # Paths for the example in Leon's thesis
    p1 = Path([G.edges[0], G.edges[1], G.edges[2]])
    p2 = Path([G.edges[3], G.edges[4], G.edges[2]])
    p3 = Path([G.edges[0], G.edges[5], G.edges[6]])
    return [p1, p2, p3]


def getNguyenPaths(G: Network, s: Node, t: Node) -> List[Path]:
    # Paths for the example in Leon's thesis
    # print(s,t,int(s.name)==1,s.name=="4",t.name=="2",t.name=="3")
    pathlist = []
    if int(s.name)==1:
        if int(t.name)==2:
            pathlist.append(Path([G.edges[1-1], G.edges[4-1], G.edges[13-1]]))
            pathlist.append(Path([G.edges[1-1], G.edges[3-1], G.edges[7-1], G.edges[8-1], G.edges[13-1]]))
            pathlist.append(Path([G.edges[1-1], G.edges[3-1], G.edges[7-1], G.edges[12-1], G.edges[16-1]]))
            pathlist.append(Path([G.edges[1-1], G.edges[3-1], G.edges[11-1], G.edges[15-1], G.edges[16-1]]))

            pathlist.append(Path([G.edges[2-1], G.edges[6-1], G.edges[7-1], G.edges[8-1], G.edges[13-1]]))
            pathlist.append(Path([G.edges[2-1], G.edges[6-1], G.edges[7-1], G.edges[12-1], G.edges[16-1]]))
            pathlist.append(Path([G.edges[2-1], G.edges[6-1], G.edges[11-1], G.edges[15-1], G.edges[16-1]]))
            pathlist.append(Path([G.edges[2-1], G.edges[10-1], G.edges[14-1], G.edges[15-1], G.edges[16-1]]))
        elif int(t.name)==3:
            pathlist.append(Path([G.edges[1-1], G.edges[3-1], G.edges[7-1], G.edges[12-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[1-1], G.edges[3-1], G.edges[11-1], G.edges[15-1], G.edges[18-1]]))

            pathlist.append(Path([G.edges[2-1], G.edges[6-1], G.edges[7-1], G.edges[12-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[2-1], G.edges[6-1], G.edges[11-1], G.edges[15-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[2-1], G.edges[10-1], G.edges[14-1], G.edges[15-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[2-1], G.edges[10-1], G.edges[17-1], G.edges[19-1]]))
    elif int(s.name)==4:
        if int(t.name)==2:
            pathlist.append(Path([G.edges[5-1], G.edges[6-1], G.edges[7-1], G.edges[8-1], G.edges[13-1]]))
            pathlist.append(Path([G.edges[5-1], G.edges[6-1], G.edges[7-1], G.edges[12-1], G.edges[16-1]]))
            pathlist.append(Path([G.edges[5-1], G.edges[6-1], G.edges[11-1], G.edges[15-1], G.edges[16-1]]))
            pathlist.append(Path([G.edges[5-1], G.edges[10-1], G.edges[14-1], G.edges[15-1], G.edges[16-1]]))

            pathlist.append(Path([G.edges[9-1], G.edges[14-1], G.edges[15-1], G.edges[16-1]]))
        elif int(t.name)==3:
            pathlist.append(Path([G.edges[5-1], G.edges[6-1], G.edges[7-1], G.edges[12-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[5-1], G.edges[6-1], G.edges[11-1], G.edges[15-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[5-1], G.edges[10-1], G.edges[14-1], G.edges[15-1], G.edges[18-1]]))

            pathlist.append(Path([G.edges[9-1], G.edges[14-1], G.edges[15-1], G.edges[18-1]]))
            pathlist.append(Path([G.edges[9-1], G.edges[17-1], G.edges[19-1]]))

    return pathlist


# def setInitialPathFlows(commodityId: int, G: Network, s: Node, t: Node,	u: PWConst,\
        # zeroflow: PartialFlow, pathInflows: PartialFlowPathBased) -> PartialFlowPathBased:
def setInitialPathFlows(G: Network, pathList : List[Path],\
        commodities : List[Tuple[Node, Node, PWConst]],\
        zeroflow: PartialFlow, pathInflows: PartialFlowPathBased) -> PartialFlowPathBased:
    print("To be implemented! Passing hardcoded path inflows.")

    for i,(s,t,u) in enumerate(commodities):
        # Get pathlist
        # pathlist = getEVExamplePaths(G, s, t)
        # pathlist = getLeonsPaths(G, s, t)
        # pathlist = getNguyenPaths(G, s, t)

        # TODO: get rid of this temporary hack asap
        if not pathList:
            pathList = getNguyenPaths(G, s, t)

        # Get flowlist
        # flowlist = [u,PWConst([0,50],[0],0),PWConst([0,50],[0],0)]
        flowlist = [PWConst([0,50],[0],0)]*(len(pathList)-1)
        flowlist.insert(0,u)
        # print("len ", len(pathlist), len(flowlist))
        # exit(0)

        # pathInflows.setPaths(commodityId, [p1,p2,p3], [u,PWConst([0,50],[0],0),PWConst([0,50],[0],0)])
        print("Setting paths up for s-t commodity: ", s, "-", t)
        pathInflows.setPaths(i, pathList, flowlist)
        # print(pathlist, flowlist)
        # print("Setting up path ", p2)
        # pathInflows.setPaths(commodityId, p2, 0)
        # print("Setting up path ", p3)
        # pathInflows.setPaths(commodityId, p3, 0)
    return pathInflows


def getAllSTpaths(G: Network, s: Node, t: Node, flow: PartialFlow) -> List[Path]:
    print("To be implemented: passing hardcoded paths for now.")
    # p1 = Path([G.edges[0], G.edges[2], G.edges[4]])
    # p2 = Path([G.edges[1], G.edges[2], G.edges[3]])
    # p3 = Path([G.edges[1], G.edges[2], G.edges[4]])
    # pathList = [p1,p2,p3]
    # print("Path list :", pathList)
    # print("Freeflow travel times :", (sum(t) for t in p1.edges.tau)
    return pathList


def fixedPointUpdate(oldPathInflows: PartialFlowPathBased, timeHorizon:
        ExtendedRational, alpha: float, timestepSize, commodities, verbose: bool) -> PartialFlowPathBased:
    currentFlow = networkLoading(oldPathInflows, timeHorizon)

    # TODO: Adjust during algorithm?
    # timestepSize = ExtendedRational(1,1)
    threshold = ExtendedRational(1,100)

    newPathInflows = PartialFlowPathBased(oldPathInflows.network, oldPathInflows.getNoOfCommodities())

    # record the difference of derived times and shortest path times
    timeDiff = [[0 for i in range(int(timeHorizon/timestepSize))] for j in
            range(oldPathInflows.getNoOfCommodities())]
    print("timeDiff ", len(timeDiff),
            range(oldPathInflows.getNoOfCommodities()),range(int(timeHorizon/timestepSize)),timeDiff)
    # exit(0)
    # for i in range(oldPathInflows.getNoOfCommodities()):
    for i,comd in enumerate(commodities):
        flowValue = [None]*len(oldPathInflows.fPlus[i])
        travelTime = [None]*len(oldPathInflows.fPlus[i])
        if verbose: print("Considering commodity ", i)
        newPathInflows.setPaths(i,[P for P in oldPathInflows.fPlus[i]],[PWConst([ExtendedRational(0,1)],[],ExtendedRational(0,1)) for P in oldPathInflows.fPlus[i]])
        s = newPathInflows.sources[i]
        t = newPathInflows.sinks[i]
        theta = ExtendedRational(0,1)
        meanIter = 0
        # We subdivide the time into intervals of length timestepSize
        k = -1
        while theta < oldPathInflows.getEndOfInflow(i):
            k += 1
            # For each subinterval i we determine the dual variable v_i
            # (and assume that it will stay the same for the whole interval)
            # if verbose: print("timeinterval [", theta, ",", theta+timestepSize,"]")

	    # Set up the update problem for each subinterval
            # print("\nSetting up the update problem for subinterval [", theta,
                    # ",", theta+timestepSize,"]\n")
            maxTravelTime = 0
            # Get path travel times for this subinterval
            for j,P in enumerate(oldPathInflows.fPlus[i]):
                 fP = oldPathInflows.fPlus[i][P]
                 # converting to float (optimize.root does not work with fractions)
                 travelTime[j] = float(currentFlow.pathArrivalTime(P,
                     theta + timestepSize/2) - (theta + timestepSize/2))
                 flowValue[j] = float(fP.getValueAt(theta))
                 maxTravelTime = max(maxTravelTime, travelTime[j])
                 print("Path: ",printPathInNetwork(P,currentFlow.network), "flowValue: ", flowValue[j], "travelTime: ",\
                         travelTime[j], "at theta =",\
                         round(float(theta + timestepSize/2),2), "fp: ", fP)

            # Compare with the shortest path travel time
            shortestPath = findShortestSTpath(s,t,currentFlow,theta + timestepSize/2)
            shortestTravelTime = currentFlow.pathArrivalTime(shortestPath,\
                    theta+timestepSize/2)- (theta + timestepSize/2)
            print("shortest path ", shortestPath.getFreeFlowTravelTime(), round(float(theta +\
                             timestepSize/2),2), printPathInNetwork(shortestPath,currentFlow.network), shortestTravelTime)
            timeDiff[i][k] = maxTravelTime - shortestTravelTime
            print("check ", i,k,len(timeDiff),timeDiff[0][0])
            if timeDiff[i][k] < 0:
                print("maxTravelTime %.2f less than shortestTravelTime %.2f!"\
                        % (float(maxTravelTime),float(shortestTravelTime)))
                # exit(0)

            # TODO: Find integral value, ubar, of (piecewise constant) function u in this
            # subinterval
            # uval1 = u.getValueAt(theta)
            # uval2 = u.getValueAt(theta + timestepSize)
            uval1 = comd[2].getValueAt(theta)
            uval2 = comd[2].getValueAt(theta + timestepSize)
            if uval1==uval2:
                ubar = uval1*timestepSize
            else:
                if not uval2 == 0:
                    print("Implement me: find integral of u when it has different\
                            positive values within a subinterval")
                    exit(0)

            # TODO: Find a good starting point
            # A trivial guess: assume all terms to be positive and solve for the dual variable
            x0 = ((-sum(flowValue) + alpha*sum(travelTime))*timestepSize +
                    ubar)/(len(flowValue)*timestepSize)
            # print("x0 ", round(x0,2))
            # optimize.show_options(solver='root', method='broyden1', disp=True)
            # TODO: Find a way to run optimize.root quietly
            # sol = optimize.root(dualVarRootFunc, x0, (alpha, flowValue, travelTime,
                    # timestepSize, ubar), method='broyden1', options={'disp':False})
            # bracketLeft = -max(list(map(float.__sub__, list(map(lambda x: alpha*x,
                # travelTime)), flowValue)))
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
            sol = optimize.root_scalar(dualVarRootFuncComb, (alpha, flowValue, travelTime,
                timestepSize, ubar), x0=x0, bracket=[bracketLeft, bracketRight],
                fprime=True, method='newton')

            # Uncomment below when using (multivariate) optimize.root() function
            # if not sol.success:
                # print("The optimize.root() method has failed with the message:")
                # print("\"", sol.message, "\"")
                # exit(0)
            # Uncomment below when using (scalar) optimize.root.scalar() function
            if not sol.converged:
                print("The optimize.root_scalar() method has failed to converge due\
                        to the following reason:")
                print("\"", sol.flag, "\"")
                exit(0)
            else:
                meanIter += sol.iterations
                # print(sol)
            # print("currentFlow ", currentFlow)
            for j,P in enumerate(oldPathInflows.fPlus[i]):
                # Uncomment below when using (multivariate) optimize.root() function
                # newFlowVal = max(flowValue[j] - alpha*travelTime[j] + sol.x, 0)
                # Uncomment below when using optimize.root_scalar() function
                newFlowVal = max(flowValue[j] - alpha*travelTime[j] + sol.root, 0)
                # print("newFlowVal ", newFlowVal)
                newPathInflows.fPlus[i][P].addSegment(ExtendedRational(theta +
                    timestepSize), ExtendedRational(newFlowVal))
            # print("newPathInflows: ", newPathInflows)
            theta = theta + timestepSize
        tmpVar = max(timestepSize,1/timestepSize)
        print("Mean # of root.scalar() iterations ",\
                float(round(meanIter/(tmpVar*oldPathInflows.getEndOfInflow(i)),2)),\
                " for ", tmpVar*oldPathInflows.getEndOfInflow(i), " subintervals")
    # for id, e in enumerate(currentFlow.network.edges):
        # print("queue at edge %d: "%id, e, currentFlow.queues[e])
    print("timeDiff ", timeDiff)
    # exit(0)
    print("newPathInflows: ", newPathInflows)
    return newPathInflows

def dualVarRootFunc(x, alpha, flowValue, travelTime, timestepSize, ubar):
    # print("printing args ", round(x,2),alpha,flowValue,travelTime,timestepSize,ubar)
    termSum = 0
    for j,fv in enumerate(flowValue):
        # print("print terms for j", j, " : ", flowValue[j],alpha,travelTime[j],x,timestepSize)
        # termSum += max(flowValue[j] - alpha*travelTime[j] + x[0], 0)*timestepSize
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
    termSum = 0
    gradTermSum = 0
    for j,fv in enumerate(flowValue):
        tmp = flowValue[j] - alpha*travelTime[j] + x
        if tmp > 0:
            termSum += tmp*timestepSize
            gradTermSum += timestepSize
    return float(termSum - ubar), float(gradTermSum)


def differenceBetweenPathInflows(oldPathInflows : PartialFlowPathBased, newPathInflows : PartialFlowPathBased) -> ExtendedRational:
    assert (oldPathInflows.getNoOfCommodities() == newPathInflows.getNoOfCommodities())
    #TODO: Also check if the time horizon for both the pathInflows is same or not

    difference = ExtendedRational(0)

    for i in range(oldPathInflows.getNoOfCommodities()):
        for path in oldPathInflows.fPlus[i]:
            if path in newPathInflows.fPlus[i]:
                # print("difference ", round(float(difference),2))
                # print("oldvals ", oldPathInflows.fPlus[i][path])
                # print("newvals ", newPathInflows.fPlus[i][path])
                # print("diff ", oldPathInflows.fPlus[i][path] + newPathInflows.fPlus[i][path].smul(ExtendedRational(-1,1)))
                difference += (oldPathInflows.fPlus[i][path] + newPathInflows.fPlus[i][path].smul(ExtendedRational(-1,1))).norm()
                # print("difference ", round(float(difference),2))
            else:
                difference += oldPathInflows.fPlus[i][path].norm()
        for path in newPathInflows.fPlus[i]:
            if path not in oldPathInflows.fPlus[i]:
                difference += newPathInflows.fPlus[i][path].norm()

    return difference

# Function arguments: (network, precision, List[source node, sink node, ?], time
# horizon, maximum allowed number of iterations, verbosity on/off)
# TODO: make provision to warm-start the script given path flow
def fixedPointAlgo(N : Network, pathList : List[Path], precision : float, commodities :
        List[Tuple[Node, Node, PWConst]], timeHorizon:
        ExtendedRational=math.inf, maxSteps: int = None, timeStep: int = None,
        alpha : float = None, verbose : bool = False) -> PartialFlowPathBased:
    step = 0

    ## Init:
    # Create zero-flow (PP: why?)
    pathInflows = PartialFlowPathBased(N,0)
    zeroflow = networkLoading(pathInflows,timeHorizon)

    pathInflows = PartialFlowPathBased(N, len(commodities))
    # Initial flow: For every commodity, select the shortest s-t path and send
    # all flow along this path (and 0 flow along all other paths)
    # for i,(s,t,u) in enumerate(commodities):
        # pathInflows.setPaths(i, [findShortestSTpath(s, t, zeroflow, ExtendedRational(0))], [u])
        # print("u ", u)
        # setInitialPathFlows(i, N, s, t, u, zeroflow, pathInflows)
        # print(pathInflows)

    setInitialPathFlows(N, pathList, commodities, zeroflow, pathInflows)

    if verbose: print("Starting with flow: \n", pathInflows)

    ## Iteration:
    while maxSteps is None or step < maxSteps:
        if verbose: print("Starting iteration #", step)
        # newpathInflows = networkLoading(pathInflows,timeHorizon)
        # print("newpathInflows ", newpathInflows)
        newpathInflows = fixedPointUpdate(pathInflows, timeHorizon, alpha,
                timeStep, commodities, verbose)
        if differenceBetweenPathInflows(pathInflows,newpathInflows) < precision:
            print("Attained required precision!")
            return newpathInflows
        if verbose: print("Changed amount is ",
                round(float(differenceBetweenPathInflows(pathInflows,newpathInflows)),2))
        # if verbose: print("Current flow is\n", newpathInflows)
        pathInflows = newpathInflows
        step += 1
        print("\nEND OF STEP ", step,"\n")

    print("Maximum number of steps reached without attaining required precision!")
    return pathInflows

