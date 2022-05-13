from networkloading import *
from fixedPointAlgorithm import *

## Nguyen Network ##

# linkData
# tail  head    capacity    Length  FreeFlowTime    OtherInfo
# 1	12	0.833	3000	150	1
# 1	5	0.833	1500	75	1
# 12	6	0.833	1500	75	1
# 12	8	0.833	9000	150	1
# 4	5	0.833	3000	150	1
# 5	6	0.833	3000	150	1
# 6	7	0.833	3000	150	1
# 7	8	0.833	3000	150	1
# 4	9	0.833	4500	225	1
# 5	9	0.833	1500	75	1
# 6	10	0.833	1500	75	1
# 7	11	0.833	1500	75	1
# 8	2	0.833	1500	75	1
# 9	10	0.833	3000	150	1
# 10	11	0.833	3000	150	1
# 11	2	0.833	3000	150	1
# 9	13	0.833	4500	225	1
# 11	3	0.833	1500	75	1
# 13	3	0.833	3000	150	1

G = Network()
# Nodes
G.addNode("1")
G.addNode("2")
G.addNode("3")
G.addNode("4")
G.addNode("5")
G.addNode("6")
G.addNode("7")
G.addNode("8")
G.addNode("9")
G.addNode("10")
G.addNode("11")
G.addNode("12")
G.addNode("13")
# Edges
G.addEdge("1", "12", ExtendedRational(5,6), ExtendedRational(150))
G.addEdge("1", "5", ExtendedRational(5,6),  ExtendedRational(75 ))
G.addEdge("12", "6", ExtendedRational(5,6), ExtendedRational(75 ))
G.addEdge("12", "8", ExtendedRational(5,6), ExtendedRational(150))
G.addEdge("4", "5", ExtendedRational(5,6),  ExtendedRational(150))
G.addEdge("5", "6", ExtendedRational(5,6),  ExtendedRational(150))
G.addEdge("6", "7", ExtendedRational(5,6),  ExtendedRational(150))
G.addEdge("7", "8", ExtendedRational(5,6),  ExtendedRational(150))
G.addEdge("4", "9", ExtendedRational(5,6),  ExtendedRational(225))
G.addEdge("5", "9", ExtendedRational(5,6,), ExtendedRational(75 ))
G.addEdge("6", "10", ExtendedRational(5,6), ExtendedRational(75 ))
G.addEdge("7", "11", ExtendedRational(5,6), ExtendedRational(75 ))
G.addEdge("8", "2", ExtendedRational(5,6),  ExtendedRational(75 ))
G.addEdge("9", "10", ExtendedRational(5,6), ExtendedRational(150))
G.addEdge("10", "11", ExtendedRational(5,6),ExtendedRational(150))
G.addEdge("11", "2", ExtendedRational(5,6), ExtendedRational(150))
G.addEdge("9", "13", ExtendedRational(5,6), ExtendedRational(225))
G.addEdge("11", "3", ExtendedRational(5,6), ExtendedRational(75 ))
G.addEdge("13", "3", ExtendedRational(5,6), ExtendedRational(150))

# Path List (link sequence)
# 1	4	13	0	0
# 1	3	7	8	13
# 1	3	7	12	16
# 1	3	11	15	16

# 2	6	7	8	13
# 2	6	7	12	16
# 2	6	11	15	16
# 2	10	14	15	16

# 1	3	7	12	18
# 1	3	11	15	18

# 2	6	7	12	18
# 2	6	11	15	18
# 2	10	14	15	18
# 2	10	17	19	0

# 5	6	7	8	13
# 5	6	7	12	16
# 5	6	11	15	16
# 5	10	14	15	16

# 9	14	15	16	0

# 5	6	7	12	18
# 5	6	11	15	18
# 5	10	14	15	18

# 9	14	15	18	0
# 9	17	19	0	0

# Source nodes: 1,4; Sink nodes: 2,3


## INPUT PARAMETERS
timeHorizon = 675    # discretization time step
maxIter = 200	    # maximum iterations of fixed point algorithm
precision = ExtendedRational(1,2)	    # desired numerical threshold for convergence
# PP: What is a good way to decide timeStep based on the given network?
# timeStep = ExtendedRational(1,4)	    # discretization time step
timeStep = ExtendedRational(1,2)	    # discretization time step
# alpha*travelTimes must be numerically comparable to pathflows [han2019]
alpha = 1	    # step size parameter

f = fixedPointAlgo(G,precision, [\
        (G.getNode("1"), G.getNode("2"),PWConst([0,10,50],[3,0],0)),\
        (G.getNode("1"), G.getNode("3"),PWConst([0,10,50],[3,0],0)),\
        (G.getNode("4"), G.getNode("2"),PWConst([0,10,50],[3,0],0)),\
        (G.getNode("4"), G.getNode("3"),PWConst([0,10,50],[3,0],0))],
        timeHorizon,maxIter,timeStep,alpha,True)
print(f)
