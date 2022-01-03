from networkloading import *
from fixedPointAlgorithm import *
import os, gzip
import numpy
import time

from typing import List
from collections import deque

# Utility function to check if current vertex is already present in path
def isNotVisited(x: Node, path: Path) -> bool:
    # print("nodes in path ", str(path.getNodesInPath()))
    if x in path.getNodesInPath():
        # print("False")
        return False
    else:
        # print("True")
        return True

# Utility function for finding paths in graph from source to destination
def findpaths(G: Network, src, dest, numNodes, EB) -> None:
    # Queue to store (partial) paths
    q = deque()

    # Add edges going out of the source to q
    for e in G.edges:
        # print(e.node_from, src)
        if e.node_from == src:
            q.append(Path([e]))
            # print("match")
        # else:
            # print("not matched")
    # print("q ", q)
    
    # List to store the final paths
    # pathList = List[Path] # PP: why does this not work?
    pathList = []
    # Path vector to store the current path
    # path = Path([], src)
    count = 0
    
    while q:
        count += 1
        print("\ncount:%d"%count)
        
        # Get the (earliest generated) partial path
        print("q before pop")
        for p in q:
            print(printPathInNetwork(p, G))
        path = q.popleft()
        print("after pop ", printPathInNetwork(path, G), q)
        for p in q:
            printPathInNetwork(p, G)
        
        # Get the last node in the partial path
        last = path.edges[-1].node_to
        print("last ", last)
        
        # If the last node is the destination node then store the path
        if last == dest:
            print("Found s-t Path:", printPathInNetwork(path, G))
            pathList.append(path)
        
        # Traverse all the nodes connected to the current node and push new partial
        # path to queue
        edgeListCurrNode = [e for e in G.edges if e.node_from == last]
        # edgeListCurrNode = []
        # for e in G.edges:
            # print("e ", e, e.node_from, last)
            # if e.node_from == last:
                # edgeListCurrNode.append(e)
        print("edgeListCurrNode ", len(edgeListCurrNode), edgeListCurrNode)
        for e in edgeListCurrNode:
            print("edge %d" %G.edges.index(e), e,\
                    printPathInNetwork(path, G), isNotVisited(e.node_to, path),\
                    path.getEnergyConsump(), e.ec, (path.getEnergyConsump() + e.ec <= EB))
            if isNotVisited(e.node_to, path) and (path.getEnergyConsump() + e.ec <= EB):
                newpath = Path(path.edges)
                print("newpath before append ", printPathInNetwork(newpath,G))
                newpath.add_edge_at_end(e)
                q.append(newpath)
                print("newpath after append ", printPathInNetwork(newpath,G))
    
    # Print pathList
    print("\nAll s-t paths are:", pathList)
    for p in pathList:
        print(printPathInNetwork(p, G))


if __name__ == "__main__":
    ## Our standard example ##
    G = Network()
    G.addNode("s")
    G.addNode("u")
    G.addNode("v")
    G.addNode("t")
    G.addEdge("s", "u", ExtendedRational(2), ExtendedRational(1), ExtendedRational(2))
    G.addEdge("s", "u", ExtendedRational(2), ExtendedRational(2), ExtendedRational(1))
    G.addEdge("u", "v", ExtendedRational(1), ExtendedRational(1), ExtendedRational(0))
    G.addEdge("v", "t", ExtendedRational(2), ExtendedRational(1), ExtendedRational(2))
    G.addEdge("v", "t", ExtendedRational(2), ExtendedRational(2), ExtendedRational(1))
    # Number of vertices
    numNodes = len(G.nodes)
    src = G.getNode("s")
    dest = G.getNode("t")
    energyBudget = 3

    # Function for finding the paths
    findpaths(G, src, dest, numNodes, energyBudget)

