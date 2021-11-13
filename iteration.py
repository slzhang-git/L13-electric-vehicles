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


def iterationStep(oldPathInflows : PartialFlowPathBased) -> PartialFlowPathBased:
    currentFlow = networkLoading(oldPathInflows)

    newPathInflows = PartialFlowPathBased(oldPathInflows.network, oldPathInflows.getNoOfCommodities())

    for i in range(oldPathInflows.getNoOfCommodities()):
        # TODO
        # newPathInflows.setPaths(i, _, _)

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


def fixedPointIteration(N : Network, precision : float, commodities : List[(Node, Node, PWConst)], timeHorizon: ExtendedRational=math.inf, maxSteps: int  = None) -> PartialFlowPathBased:
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

    ## Iteration:
    while maxSteps is None or step < maxSteps:
        newpathInflows = iterationStep(pathInflows)
        if differenceBetweenPathInflows(pathInflows,newpathInflows) < precision:
            return newpathInflows
        pathInflows = newpathInflows

    print("Maximum number of steps reached without attaining required precision!")
    return pathInflows

