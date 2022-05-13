from networkloading import *
from fixedPointAlgorithm import *

## Our standard example ##
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
#
#
# p1 = Path([G.edges[0], G.edges[2], G.edges[4]])
# p2 = Path([G.edges[1], G.edges[2], G.edges[3]])
# p3 = Path([G.edges[1], G.edges[2], G.edges[4]])
#

## INPUT PARAMETERS
timeHorizon = 60    # discretization time step
maxIter = 100	    # maximum iterations of fixed point algorithm
precision = 1/2	    # desired numerical threshold for convergence
# PP: What is a good way to decide timeStep based on the given network?
timeStep = ExtendedRational(1,4)	    # discretization time step
# timeStep = ExtendedRational(1,1)	    # discretization time step
# alpha*travelTimes must be numerically comparable to pathflows [han2019]
alpha = 1	    # step size parameter

f = fixedPointAlgo(G,precision,[(G.getNode("s"),G.getNode("t"),PWConst([0,10,50],[3,0],0))],timeHorizon,maxIter,timeStep,alpha,True)
print(f)
