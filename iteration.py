from typing import List, Dict, Tuple

from networkloading import *
from dynamic_dijkstra import dynamic_dijkstra


def findShortestSTpath(s: Node, t: Node, flow: PartialFlow, time: ExtendedRational) -> Path:
    (arrivalTimes, realizedTimes) = dynamic_dijkstra(time, s, t, flow)
    p = Path([], t)
    while p.getStart() != s:
        v = p.getStart()
        for e in v.incoming_edges:
            if arrivalTimes[e.node_from] + realizedTimes[e] == arrivalTimes[v]:
                p.add_edge_at_start(e)
                break

    return p


def iterationStepNaive(oldPathInflows : PartialFlowPathBased, verbose : bool) -> PartialFlowPathBased:
    currentFlow = networkLoading(oldPathInflows)

    # TODO: Adjust during algorithm?
    timestepSize = ExtendedRational(1)
    threshold = 0

    newPathInflows = PartialFlowPathBased(oldPathInflows.network, oldPathInflows.getNoOfCommodities())

    for i in range(oldPathInflows.getNoOfCommodities()):
        if verbose: print("Considering commodity ", i)
        for P in oldPathInflows.fPlus[i]:
            newPathInflows.fPlus[i][P] = PWConst([ExtendedRational(0)],[],ExtendedRational(0))
        s = oldPathInflows.sources[i]
        t = oldPathInflows.sinks[i]
        theta = ExtendedRational(0)
        # We subdivide the time into intervals of length timestepSize
        while theta < oldPathInflows.getEndOfInflow(i):
            # For each such interval we determine the shortest path at the beginning of the interval
            # (and basically assume it will stay the shortest one for the whole interval)
            if verbose: print("timeinterval [", theta, ",", theta+timestepSize,"]")
            shortestPath = findShortestSTpath(s,t,currentFlow,theta)
            if shortestPath not in newPathInflows.fPlus[i]:
                newPathInflows.fPlus[i][shortestPath] = PWConst([ExtendedRational(0)], [], ExtendedRational(0))
            shortestTravelTime = currentFlow.pathArrivalTime(shortestPath,theta)
            # We then go through all the paths used in the old flow distribution and check whether they are longer
            # then the currently shortest path (+ some threshold)
            redistributedFlow = PWConst([theta, theta+timestepSize],[ExtendedRational(0)],default=ExtendedRational(0))
            for P in oldPathInflows.fPlus[i]:
                fP = oldPathInflows.fPlus[i][P]
                if currentFlow.pathArrivalTime(P,theta) > shortestTravelTime + threshold:
                    # If so we will take all flow from this path away and redistribute it to the shortest path
                    redistributedFlow += fP.restrictTo(theta,theta+timestepSize)
                    if verbose: print("redistributing flow from ", P, " to ", shortestPath)
                    print(redistributedFlow)
                else:
                    newPathInflows.fPlus[i][P] += fP.restrictTo(theta,theta+timestepSize)
            newPathInflows.fPlus[i][shortestPath] += redistributedFlow
            print(newPathInflows.fPlus[i][shortestPath])
            theta = theta+timestepSize

    return newPathInflows


def differenceBetweenPathInflows(oldPathInflows : PartialFlowPathBased, newPathInflows : PartialFlowPathBased) -> ExtendedRational:
    assert (oldPathInflows.getNoOfCommodities() == newPathInflows.getNoOfCommodities())
    difference = ExtendedRational(0)

    for i in range(oldPathInflows.getNoOfCommodities()):
        for path in oldPathInflows.fPlus[i]:
            if path in newPathInflows.fPlus[i]:
                difference += (oldPathInflows.fPlus[i][path] + newPathInflows.fPlus[i][path].smul(ExtendedRational(-1,1))).norm()
            else:
                difference += oldPathInflows.fPlus[i][path].norm()
        for path in newPathInflows.fPlus[i]:
            if path not in oldPathInflows.fPlus[i]:
                difference += newPathInflows.fPlus[i][path].norm()

    return difference


def fixedPointIteration(N : Network, precision : float, commodities : List[Tuple[Node, Node, PWConst]], timeHorizon: ExtendedRational=math.inf, maxSteps: int = None, verbose : bool = False) -> PartialFlowPathBased:
    step = 0

    ## Init:
    # Create zero-flow
    pathInflows = PartialFlowPathBased(N,0)
    zeroflow = networkLoading(pathInflows,timeHorizon)

    i = 0
    pathInflows = PartialFlowPathBased(N, len(commodities))
    # Initial flow: For every commodity select one source sink path and send all flow along this path
    for (s,t,u) in commodities:
        pathInflows.setPaths(i, [findShortestSTpath(s, t, zeroflow, ExtendedRational(0))], [u])
        i += 1

    if verbose: print("Starting with flow: \n", pathInflows)

    ## Iteration:
    while maxSteps is None or step < maxSteps:
        if verbose: print("Starting iteration #", step)
        newpathInflows = iterationStepNaive(pathInflows, verbose)
        if differenceBetweenPathInflows(pathInflows,newpathInflows) < precision:
            return newpathInflows
        if verbose: print("Changed amount is ", differenceBetweenPathInflows(pathInflows,newpathInflows))
        pathInflows = newpathInflows

    print("Maximum number of steps reached without attaining required precision!")
    return pathInflows

