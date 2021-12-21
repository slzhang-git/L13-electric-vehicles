#!/usr/bin/python

# TODO: Use getopt or argparse to read command line arguments
import sys, time
from networkloading import *
from fixedPointAlgorithm import *

## Leon's example network ##
def genLeonNetwork():
    G = Network()
    G.addNode("s")
    G.addNode("u")
    G.addNode("v")
    G.addNode("t")
    G.addNode("b")
    G.addNode("a")
    G.addEdge("s", "u", ExtendedRational(3,2), ExtendedRational(2))
    G.addEdge("u", "v", ExtendedRational(1), ExtendedRational(2))
    G.addEdge("v", "t", ExtendedRational(1,2), ExtendedRational(2))
    G.addEdge("s", "a", ExtendedRational(3), ExtendedRational(3))
    G.addEdge("a", "v", ExtendedRational(3), ExtendedRational(3))
    G.addEdge("u", "b", ExtendedRational(3), ExtendedRational(3))
    G.addEdge("b", "t", ExtendedRational(3), ExtendedRational(3))

    return G

def getLeonsPaths(G: Network, s: Node, t: Node) -> List[Path]:
    # Paths for the example in Leon's thesis
    p1 = Path([G.edges[0], G.edges[1], G.edges[2]])
    p2 = Path([G.edges[3], G.edges[4], G.edges[2]])
    p3 = Path([G.edges[0], G.edges[5], G.edges[6]])
    return [p1, p2, p3]



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
    arglist = []
    for i in range(1,n):
        arglist.append(argv[i])
        # print(i,arglist[i-1])
    # for i in range(len(arglist)):
        # if arglist[i] not in exclude:
            # args.append(arglist[i])
            # print(i,arglist[i])
    return arglist

         
if __name__ == "__main__":
    # Read arguments into required variables
    [insName,timeHorizon,maxIter,precision,alpha,timeStep] = readArgs(sys.argv)
    print("read as: insName,timeHorizon,maxIter,precision,alpha,timeStep")
    print("check: ",insName,timeHorizon,maxIter,precision,alpha,timeStep)

    if insName == "leon":
        G = genLeonNetwork()
        pathList = getLeonsPaths(G,G.getNode("s"),G.getNode("t"))
    elif insName == "evExample3":
        G = genEvExample3Network()
    elif insName == "Nguyen":
        G = genNguyenNetwork()
    else:
        print("Unknown network name. Exiting.")
        exit(0)

    
    tStart = time.time()
    # filename = 'alpha=%.2f'%alpha + '_maxIter=%d'%maxIter +\
    # '_timeStep=%.2f'%timeStep + '_precision=%.2f'%precision + '.npz'
    # print("------------------------------------------------\n",filename,\
            # "\n------------------------------------------------")
    f = fixedPointAlgo(G,pathList,float(precision),[(G.getNode("s"),G.getNode("t"),\
            PWConst([0,10,50],[3,0],0))],ExtendedRational(timeHorizon),\
            int(maxIter),ExtendedRational(timeStep),ExtendedRational(alpha),True)
    tEnd = time.time()
    # print(f)
    print("Elasped wall time: ", round(tEnd-tStart,4))
    # numpy.savez(filename, f=f,time=tEnd-tStart)

