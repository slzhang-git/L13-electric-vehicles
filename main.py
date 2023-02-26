#!/usr/bin/python

# TODO: Use getopt or argparse to read command line arguments
import sys, time, os, numpy
from networkloading import *
from fixedPointAlgorithm import *

# For reading graphs
import networkx as nx

args_manual=['main.py', 'evExample6EdgesWR.txt', 'evExample6Comm.txt', 'example6', '50', '20000', '3600', '0.001', '0.75', '0.25', '0', '1']

def readArgs(argv):
    # Arguments passed
    print("\nName of script:", argv[0])
    n = len(argv)
    print("Total arguments passed:", n)
    print(argv)

    # Read the instance name
    insName = argv[0]

    # exclude = {","," ","[","]"}
    argList = []
    for i in range(1,n):
        argList.append(argv[i])
    return argList


def readNetwork(edgeList, verbose: bool=False) -> Network:
    #TODO: put checks for a valid network
    # First read as a MultiDiGraph
    Gn = nx.MultiDiGraph()
    # Reading ExtendedRational here to allow input data in fractions (e.g. 5/6)
    Gn = nx.read_edgelist(edgeList, comments='#', nodetype=str,\
            create_using=nx.MultiDiGraph, data=(("nu", ExtendedRational),\
            ("tau", ExtendedRational), ("ec", ExtendedRational),\
            ("price", ExtendedRational),))
    if verbose: print('edges: ', list(Gn.edges(data=True)))
    if verbose: print('nodes: ', list(Gn.nodes()))

    # Convert to a Network object for our purposes
    G = Network()
    for node in Gn.nodes():
        G.addNode(node)
    # Converting to required data type (ExtendedRational or Float)
    for u,v,data in Gn.edges(data=True):
        G.addEdge(u,v,makeNumber(data['nu']), makeNumber(data['tau']),
                makeNumber(data['ec']), makeNumber(data['price']))

    #TODO: Plot the graph using nx
    return G


def readCommodities(commList) -> List[Tuple[Node, Node, PWConst]]:
    commodities = []
    timesStartPos = 4
    with open(commList, 'r') as fobj:
        for line in fobj:
            # print('line ', line)
            line = line.partition('#')[0]
            line = line.rstrip()
            if line:
                data = [entry for entry in line.split()]
                # Create the PWConst function for this commodity
                times = [makeNumber(i) for i in data[timesStartPos:timesStartPos +\
                        math.ceil((len(data)-timesStartPos)/2)]]
                vals = [makeNumber(i) for i in data[timesStartPos+len(times):len(data)]]
                # The third argument = 0 means that the inflow rate is 0 for the rest of
                # the real line outside the specified time intervals
                pwcf = PWConst(times, vals, 0)
                commodities.append((G.getNode(data[0]), G.getNode(data[1]),
                    makeNumber(data[2]), makeNumber(data[3]), pwcf))
    return commodities


if __name__ == "__main__":

    argList = readArgs(args_manual)

    G = readNetwork(argList[0])
    nuMin, nuMax = min([e.nu for e in G.edges]), max([e.nu for e in G.edges])
    tauMin, tauMax = min([e.tau for e in G.edges]), max([e.tau for e in G.edges])
    ecMin, ecMax = min([e.ec for e in G.edges]), max([e.ec for e in G.edges])
    priceMin, priceMax = min([e.price for e in G.edges]), max([e.price for e in G.edges])
    if True: print('Min.: nu = %.2f, tau = %.2f, ec = %.2f, price = %.2f'%(round(float(nuMin),2),
        round(float(tauMin),2), round(float(ecMin),2), round(float(priceMin),2)))
    if True: print('Max.: nu = %.2f, tau = %.2f, ec = %.2f, price = %.2f'%(round(float(nuMax),2),
        round(float(tauMax),2), round(float(ecMax),2), round(float(priceMax),2)))

    commodities = readCommodities(argList[1])

    fname = ""
    for i in range(2,len(argList)):
        fname += argList[i] + "_"

    # Read arguments into required variables
    [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,priceToTime,numThreads] = argList[2:len(argList)]
    [insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,priceToTime,numThreads] = [str(insName),\
            makeNumber(timeHorizon),int(maxIter),int(timeLimit),float(precision),\
            makeNumber(alpha),makeNumber(timeStep),makeNumber(priceToTime),int(numThreads)]
    print("read args: insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,priceToTime,numThreads")
    print("values: ",insName,timeHorizon,maxIter,timeLimit,precision,alpha,timeStep,priceToTime,numThreads)

    # Find list of paths for each commodity
    # TODO: put data checks
    pathList = []
    tStart = time.time()
    for i,(s,t,energyBudget,priceBudget,u) in enumerate(commodities):
        if True: print("\nFinding paths for comm %d: %s-%s"%(i,s,t),energyBudget,priceBudget,u)
        # pathList.append(G.findPaths(s, t, energyBudget))
        paths = G.findPathsWithLoops(s, t, energyBudget, priceBudget, numThreads)
        if len(paths) > 0:
            pathList.append(paths)
        else:
            print('No feasible paths found for comm %d: '%i, s,t,energyBudget,priceBudget,u)
            exit(0)
        # for j,P in enumerate(paths):
            # print(P)
             # print("path%d"%j, G.printPathInNetwork(P), ": energy cons.: ",
                     # P.getNetEnergyConsump(), ": latency: ",P.getFreeFlowTravelTime())
    print("\nTime taken in finding paths: ", round(time.time()-tStart,4))

    if True: print('Total number of paths: ', sum(len(x) for x in pathList))
    minTravelTime = infinity
    maxTravelTime = infinity*(-1)
    for p in pathList:
        maxval,minval = max([i.getFreeFlowTravelTime() for i in p]),\
        min([i.getFreeFlowTravelTime() for i in p])

        maxTravelTime,minTravelTime = max(maxTravelTime, maxval),min(minTravelTime,
                minval)

    if True: print('Max., min. path travel time: %.2f, %.2f ' %(maxTravelTime, minTravelTime))

    # Start
    tStart = time.time()
    f, alphaIter, absDiffBwFlowsIter, relDiffBwFlowsIter, travelTime, stopStr,\
            alphaStr, qopiIter, qopiFlowIter, qopiPathComm, totDNLTime, totFPUTime =\
            fixedPointAlgo(G, pathList, precision, commodities, timeHorizon,\
            maxIter, timeLimit, timeStep, alpha, priceToTime, True)

    tEnd = time.time()
    # for i,(s,t,eb,pb,u) in enumerate(commodities):
        # print("comm ", i,s,t,eb,pb,u)
        # for tt in travelTime[i]:
            # print([round(float(a),4) for a in tt])
    eventualFlow = networkLoading(f)
    # print("eventualFlow: ", eventualFlow)
    # print("Number of paths in f: ", sum([len(f.fPlus[i]) for i in
        # f.noOfCommodities]))
    print("f: ", f)
    # print("queue at: ")
    # for id, e in enumerate(eventualFlow.network.edges):
        # if eventualFlow.queues[e].noOfSegments > 1 or\
        # (eventualFlow.queues[e].noOfSegments == 1 and\
        # (eventualFlow.queues[e].segmentTvalues[0] > 0 or\
            # eventualFlow.queues[e].segmentMvalues[0] > 0)):
            # print("edge %d: "%id, e, eventualFlow.queues[e])

    # alpha and flowDiff
    ralphaIter = [round(float(a),4) for a in alphaIter]
    # ralphaIter = [[round(float(a),4) for a in b] for b in alphaIter]
    rAbsDiffBwFlowsIter = [round(float(b),4) for b in absDiffBwFlowsIter]
    rRelDiffBwFlowsIter = [round(float(b),4) for b in relDiffBwFlowsIter]
    rqopiIter = [round(float(b),4) for b in qopiIter]
    # rqopiMeanIter = [round(float(b),4) for b in qopiMeanIter]
    rqopiFlowIter = [round(float(b),4) for b in qopiFlowIter]
    print("\nalphaMean ", ralphaIter)
    print("\nabsDiffBwFlowsIter ", rAbsDiffBwFlowsIter)
    print("\nrelDiffBwFlowsIter ", rRelDiffBwFlowsIter)
    print("\nqopiIter ", rqopiIter)
    # print("\nqopiMeanIter ", rqopiMeanIter)
    print("\nqopiFlowIter ", rqopiFlowIter)

    print("\nTermination message: ", stopStr)
    print("\nAttained DiffBwFlows (abs.): ", rAbsDiffBwFlowsIter[-2])
    print("Attained DiffBwFlows (rel.): ", rRelDiffBwFlowsIter[-2])
    print("\nAttained QoPI (abs.): ", rqopiIter[-2])
    # print("Attained QoPI (mean): ", rqopiMeanIter[-2])
    print("Attained QoPI (per unit flow): ", rqopiFlowIter[-2])
    print("\nIterations : ", len(ralphaIter))
    print("\nMean time for DNL : ", round(totDNLTime/len(ralphaIter),4))
    print("Mean time for FP Update : ", round(totFPUTime/len(ralphaIter),4))
    print("\nElapsed wall time: ", round(tEnd-tStart,4))

    # Save the results to files
    # dirname = os.path.expanduser('./npzfiles')
    dirname=os.getcwd()+'/miscfiles'
    outputfile=os.path.join(dirname, fname)
    if not(os.path.exists(outputfile)):
        os.makedirs(outputfile)
    # Uncomment below to include a string indicating the alpha-update strategy
    # default is alphaSmooth(gamma)
    # fname += alphaStr.replace('/','By')
    numpy.savez(outputfile,G=G,commodities=commodities,f=f,\
            eventualFlow=eventualFlow,time=tEnd-tStart,\
            alphaIter=alphaIter,absDiffBwFlowsIter=absDiffBwFlowsIter,\
            relDiffBwFlowsIter=relDiffBwFlowsIter,travelTime=travelTime,\
            # stopStr=stopStr,alphaStr=alphaStr,qopiIter=qopiIter,qopiMeanIter=qopiMeanIter,\
            stopStr=stopStr,alphaStr=alphaStr,qopiIter=qopiIter,\
            qopiFlowIter=qopiFlowIter,qopiPathComm=qopiPathComm)
    print("\noutput saved to file: %s.npz"%os.path.join(dirname, fname))

