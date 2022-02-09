#!/usr/bin/python

# TODO: Use getopt or argparse to read command line arguments
import sys, time, os, numpy
from networkloading import *
from fixedPointAlgorithm import *

# For reading graphs
import networkx as nx

def readArgs(argv):
    # Arguments passed
    print("\nName of script:", sys.argv[0])
    n = len(sys.argv)
    print("Total arguments passed:", n)
    # print("\nArguments passed:", end = " ")
    # for i in range(1, n):
        # print(sys.argv[i], end = " ")
    # print()
    print(sys.argv)

    # Read the instance name
    insName = argv[0]

    # arglist.append(argv[i] for i in range(1,n))
    # exclude = {","," ","[","]"}
    argList = []
    for i in range(1,n):
        argList.append(argv[i])
        # print(i,arglist[i-1])
    # for i in range(len(arglist)):
        # if arglist[i] not in exclude:
            # args.append(arglist[i])
            # print(i,arglist[i])
    return argList


def readNetwork(edgeList, verbose: bool=False) -> Network:
    #TODO: put checks for a valid network
    # First read as a MultiDiGraph
    Gn = nx.MultiDiGraph()
    # Reading ExtendedRational here to allow input data in fractions (e.g. 5/6)
    Gn = nx.read_edgelist(edgeList, comments='#', nodetype=str,\
            create_using=nx.MultiDiGraph, data=(("nu", ExtendedRational),\
            ("tau", ExtendedRational), ("ec", ExtendedRational),))
    if verbose: print('edges: ', list(Gn.edges(data=True)))
    if verbose: print('nodes: ', list(Gn.nodes()))

    # Convert to a Network object for our purposes
    G = Network()
    for node in Gn.nodes():
        G.addNode(node)
    # Converting to required data type (ExtendedRational or Float)
    for u,v,data in Gn.edges(data=True):
        G.addEdge(u,v,makeNumber(data['nu']), makeNumber(data['tau']),
                makeNumber(data['ec']))

    #TODO: Plot the graph using nx
    return G


def readCommodities(commList) -> List[Tuple[Node, Node, PWConst]]:
    commodities = []
    with open(commList, 'r') as fobj:
        for line in fobj:
            # print('line ', line)
            line = line.partition('#')[0]
            line = line.rstrip()
            if line:
                data = [entry for entry in line.split()]
                # print('data ', data, len(data)-2, data[2:len(data)-1])
                # Create the PWConst function for this commodity
                times = [makeNumber(i) for i in data[2:2+math.ceil(len(data)/2 -1)]]
                vals = [makeNumber(i) for i in data[2+len(times):len(data)]]
                # print(times, vals)
                # The third argument = 0 means that the inflow rate is 0 for the rest of
                # the real line outside the specified time intervals
                pwcf = PWConst(times, vals, 0)
                # print(pwcf)
                commodities.append((G.getNode(data[0]), G.getNode(data[1]), pwcf))
    # print('comm: ', commodities)
    return commodities


if __name__ == "__main__":

    argList = readArgs(sys.argv)

    G = readNetwork(argList[0])
    nuMin, nuMax = min([e.nu for e in G.edges]), max([e.nu for e in G.edges])
    tauMin, tauMax = min([e.tau for e in G.edges]), max([e.tau for e in G.edges])
    ecMin, ecMax = min([e.ec for e in G.edges]), max([e.ec for e in G.edges])
    if True: print('Min.: nu = %.2f, tau = %.2f, ec = %.2f'%(round(float(nuMin),2),
        round(float(tauMin),2), round(float(ecMin),2)))
    if True: print('Max.: nu = %.2f, tau = %.2f, ec = %.2f'%(round(float(nuMax),2),
        round(float(tauMax),2), round(float(ecMax),2)))

    commodities = readCommodities(argList[1])

    fname = ""
    for i in range(2,len(argList)):
        fname += argList[i] + "_"
    # fname += argList[-1]

    # Read arguments into required variables
    [insName,timeHorizon,maxIter,precision,alpha,timeStep,energyBudget] = argList[2:len(argList)]
    [insName,timeHorizon,maxIter,precision,alpha,timeStep,energyBudget] = [str(insName),\
            makeNumber(timeHorizon),int(maxIter),float(precision),\
            makeNumber(alpha),makeNumber(timeStep),makeNumber(energyBudget)]
    print("read args: insName,timeHorizon,maxIter,precision,alpha,timeStep,energyBudget")
    print("values: ",insName,timeHorizon,maxIter,precision,alpha,timeStep,energyBudget)

    # Find list of paths for each commodity
    # TODO: put data checks
    pathList = []
    for i,(s,t,u) in enumerate(commodities):
        if False: print("i ", i,s,t,u)
        # pathList.append(G.findPaths(s, t, energyBudget))
        pathList.append(G.findPathsWithLoops(s, t, energyBudget))
        # for p in pathList:
            # print(p)

    if True: print('Total number of paths: ', sum(len(x) for x in pathList))
    minTravelTime = infinity
    maxTravelTime = infinity*(-1)
    for p in pathList:
        maxval,minval = max([i.getFreeFlowTravelTime() for i in p]),\
        min([i.getFreeFlowTravelTime() for i in p])

        maxTravelTime,minTravelTime = max(maxTravelTime, maxval),min(minTravelTime,
                minval)

    if True: print('Max. (min.) path travel time: %.2f (%.2f) ' %(maxTravelTime, minTravelTime))

    # Start
    tStart = time.time()
    f, alphaIter, absDiffBwFlowsIter, relDiffBwFlowsIter, travelTime, stopStr,\
            alphaStr, qopiIter = fixedPointAlgo(G, pathList, precision, commodities,\
                    timeHorizon, maxIter, timeStep, alpha, True)

    tEnd = time.time()
    print("travelTimes: ", travelTime)
    eventualFlow = networkLoading(f)
    print("eventualFlow: ", eventualFlow)
    print("f: ", f)
    print("queue at: ")
    for id, e in enumerate(eventualFlow.network.edges):
        if eventualFlow.queues[e].noOfSegments > 1 or\
        (eventualFlow.queues[e].noOfSegments == 1 and
                eventualFlow.queues[e].segmentValues[0] > 0):
            print("edge %d: "%id, e, eventualFlow.queues[e])

    # alpha and flowDiff
    ralphaIter = [round(float(b),3) for b in alphaIter]
    rAbsDiffBwFlowsIter = [round(float(b),3) for b in absDiffBwFlowsIter]
    rRelDiffBwFlowsIter = [round(float(b),3) for b in relDiffBwFlowsIter]
    rqopiIter = [round(float(b),3) for b in qopiIter]
    print("alpha ", ralphaIter)
    print("absDiffBwFlowsIter ", rAbsDiffBwFlowsIter)
    print("relDiffBwFlowsIter ", rRelDiffBwFlowsIter)
    print("qopiIter ", rqopiIter)

    print("Termination message: ", stopStr)
    print("\nElasped wall time: ", round(tEnd-tStart,4))

    # Save the results to files
    # dirname = os.path.expanduser('./npzfiles')
    dirname = os.path.expanduser('./miscfiles')
    fname += alphaStr.replace('/','By')
    numpy.savez(os.path.join(dirname, fname),G=G,f=f,eventualFlow=eventualFlow,time=tEnd-tStart,\
            alphaIter=alphaIter,absDiffBwFlowsIter=absDiffBwFlowsIter,\
            relDiffBwFlowsIter=relDiffBwFlowsIter,travelTime=travelTime,\
            stopStr=stopStr,alphaStr=alphaStr,qopiIter=qopiIter)
    print("output saved to file: %s.npz"%os.path.join(dirname, fname))

