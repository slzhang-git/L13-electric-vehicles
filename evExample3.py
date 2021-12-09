from networkloading import *
from fixedPointAlgorithm import *

## Our standard example ##
G = Network()
G.addNode("s")
G.addNode("u")
G.addNode("v")
G.addNode("t")
G.addEdge("s", "u", ExtendedRational(2), ExtendedRational(1))
G.addEdge("s", "u", ExtendedRational(2), ExtendedRational(2))
G.addEdge("u", "v", ExtendedRational(1), ExtendedRational(1))
G.addEdge("v", "t", ExtendedRational(2), ExtendedRational(1))
G.addEdge("v", "t", ExtendedRational(2), ExtendedRational(2))
#
#
p1 = Path([G.edges[0], G.edges[2], G.edges[4]])
p2 = Path([G.edges[1], G.edges[2], G.edges[3]])
p3 = Path([G.edges[1], G.edges[2], G.edges[4]])
#
# times = []
#
# lambdas = [ExtendedRational(x)
#            for x in [2+1.0/2**i for i in range(len(times)-1)]]
#
# times = [ExtendedRational(0, 1), ExtendedRational(1, 1), ExtendedRational(
#     7, 5), ExtendedRational(7, 5)+ExtendedRational(1, 1000), ExtendedRational(100, 1)]
# lambdas = [ExtendedRational(3, 1), ExtendedRational(
#     5, 2), 2+ExtendedRational(2, 9), 2+ExtendedRational(2, 9)]
#
# f1 = PWConst(times, [l for l in lambdas], ExtendedRational(0))
# f2 = PWConst(times, [ExtendedRational(3-l)
#                      for l in lambdas], ExtendedRational(0))
#
# pathInflowRates = PartialFlowPathBased(G,1)
# pathInflowRates.setPaths(0, [p1, p2], [f1, f2])
#
# flow = networkLoading(pathInflowRates)
# #flow = networkLoading(pathInflowRates, ExtendedRational(50), verbose=True)
#
#
# print(flow)
#
# for theta in times[:-1]:
#     print("Starting at ", theta, " along path P1: ", flow.pathArrivalTime(
#         p1, theta), " ≈ ", float(flow.pathArrivalTime(p1, theta)))
#     print("Starting at ", theta, " along path P2: ", flow.pathArrivalTime(
#         p2, theta), " ≈ ", float(flow.pathArrivalTime(p2, theta)))
#     print("Starting at ", theta, " along path P3: ", flow.pathArrivalTime(
#         p3, theta), " ≈ ", float(flow.pathArrivalTime(p3, theta)))
#
# theta = ExtendedRational(7, 5) + ExtendedRational(1, 100)
# print("Arrival time at u over e_1 = ",
#       flow.pathArrivalTime(Path([G.edges[0]]), theta))
# print("Arrival time at v over e_1 = ", flow.pathArrivalTime(
#     Path([G.edges[2]]), flow.pathArrivalTime(Path([G.edges[0]]), theta)))
# print("Arrival time at u over e_2 = ",
#       flow.pathArrivalTime(Path([G.edges[1]]), theta))
# print("Arrival time at v over e_2 = ", flow.pathArrivalTime(
#     Path([G.edges[2]]), flow.pathArrivalTime(Path([G.edges[1]]), theta)))
#
# # Check feasibility
# print(flow.checkFeasibility(10))

# flow.fPlus[G.edges[0], 0].drawGraph(0, 3).show()
# flow.fPlus[G.edges[1], 1].drawGraph(0, 3).show()
#flow.queues[G.edges[0]].drawGraph(0, 5).show()
#flow.queues[G.edges[2]].drawGraph(0, 5).show()
#flow.fMinus[(G.edges[2],0)].drawGraph(0, 10).show()

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
# networkLoading(f).fPlus[G.edges[0],0].drawGraph(0,10).show()
# networkLoading(f).fPlus[G.edges[2],1].drawGraph(0,10).show()
# networkLoading(f).queues[G.edges[0]].drawGraph(0,10).show()
# networkLoading(f).queues[G.edges[4]].drawGraph(0,10).show()
