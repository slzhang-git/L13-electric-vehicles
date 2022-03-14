from __future__ import annotations

import random
from typing import Union, List

from utilities import *
from collections import deque
import itertools

# A directed edge (from https://github.com/Schedulaar/predicted-dynamic-flows/blob/main/predictor/src/core/graph.py)
# with capacity nu, travel time tau, and energy consumption ec
class Edge:
    node_from: Node
    node_to: Node
    tau: number
    nu: number
    ec: number

    def __init__(self, node_from: Node, node_to: Node, capacity: number=1,
            traveltime: number=1, energyCons: number=0):
        # Creating an edge from node_from to node_to
        self.node_from = node_from
        self.node_to = node_to

        # Free flow travel time over this edge
        assert(traveltime >= 0)
        self.tau = traveltime

        # Capacity of this edge
        assert(capacity >= 0)
        self.nu = capacity

        # Energy consumption over this edge (can be negative)
        self.ec = energyCons

    def __str__(self):
        return "("+str(self.node_from)+","+str(self.node_to)+")"

    # Get edge (id) in network
    def getIdInNetwork(self, G: Network):
        return "Implement me."

# A node (from https://github.com/Schedulaar/predicted-dynamic-flows/blob/main/predictor/src/core/graph.py )
class Node:
    name: str
    # TODO: rename the parameter name id
    id: int
    incoming_edges: List[Edge]
    outgoing_edges: List[Edge]

    def __init__(self, name: str, id:int):
        # Create a node with name name and without incoming or outgoing edges
        self.name = name
        self.id = id
        self.incoming_edges = []
        self.outgoing_edges = []

    def __str__(self):
        # Print the nodes name
        if self.name == "":
            return str(self.id)
        else:
            return str(self.name)

# A network consisting of a directed graph with capacities and travel times on all edges
class Network:
    edges: List[Edge]
    nodes: List[Node]
    idCounter: int

    def __init__(self):
        # Create an empty network
        self.edges = []
        self.nodes = []
        self.idCounter = 0

    def getNode(self,node: Union[str,int,Node]) -> Union[Node,None]:
        if isinstance(node,Node):
            return node
        elif isinstance(node, str):
            for v in self.nodes:
                if v.name == node:
                    return v
        elif isinstance(node,int):
            # TODO: Schnelleres Verfahren fur Suche nach IDs?
            for v in self.nodes:
                if v.id == node:
                    return v

        return None

    def addNode(self,name: str="") -> Node:
        self.nodes.append(Node(name,self.idCounter))
        self.idCounter += 1
        return self.nodes[-1]

    def addEdge(self,node_from: Union[str,int,Node], node_to: Union[str,int,Node],
            nu: number, tau: number, ec: number=zero):
        v = self.getNode(node_from)
        w = self.getNode(node_to)
        e = Edge(v, w, nu, tau, ec)
        v.outgoing_edges.append(e)
        w.incoming_edges.append(e)
        self.edges.append(e)

    def removeEdge(self,edge:Edge):
        edge.node_to.incoming_edges.remove(edge)
        edge.node_from.outgoing_edges.remove(edge)
        self.edges.remove(edge)

    def subdivide(self,edge:Edge, nu: number, tau: number,
            ec: number):
        self.edges.remove(edge)
        v = self.addNode()
        self.addEdge(edge.node_from, v, nu, tau, ec)
        self.addEdge(v, edge.node_to, nu, tau, ec)

    def duplicate(self, edge:Edge, nu: number, tau: number, ec: number):
        self.addEdge(edge.node_from, edge.node_to, nu, tau, ec)

    def __str__(self) -> str:
        s = "Graph with " + str(len(self.nodes)) + " nodes and " + str(len(self.edges)) + " edges:\n"
        for v in self.nodes:
            s += str(v) + ": {"
            for e in v.outgoing_edges:
                s += str(e) + ", "
            s = s[:-2]
            if len(v.outgoing_edges) > 0:
                s += "}\n"
            else:
                s += "\n"
        return s

    # Print path p based on edge ids in the network G
    def printPathInNetwork(self, p: Path):
        s = str()
        for e in p.edges:
            if len([i for i in e.node_from.outgoing_edges\
                    if i.node_to == e.node_to]) > 1:
                s += str(self.edges.index(e))
            s += str(e)
        return s

    # Join two (sub)paths
    def joinPaths(self, p1: Path, p2: Path) -> Path:
        if p1.getEnd() == p2.getStart():
            path = Path(p1.edges)
            for e in p2.edges:
                path.add_edge_at_end(e)
                # print('edge: ', e, end=' ')
                # print('path: ', self.printPathInNetwork(path))
            return path
        else:
            print('Unable to join paths: ', self.printPathInNetwork(p1),
                    self.printPathInNetwork(p2))
            exit(0)
            return None

    # Concatenate two sets of (sub)paths while checking feasibility
    def joinFeasiblePaths(self, P1: List[Path], P2: List[Path], EB: number=infinity) -> List[Path]:
        pathList = []
        for p1 in P1:
            # print('for p1: ', self.printPathInNetwork(p1))
            for p2 in P2:
                # if first edge is a self loop, join (p1,p2)
                if p2.edges[0].node_from == p2.edges[0].node_to:
                    pathList.append(self.joinPaths(p1, p2))
                # else check feasibility
                else:
                    if p1.getEnergyConsump() + p2.getEnergyConsump() <= EB:
                        pathList.append(self.joinPaths(p1, p2))
        return pathList

    # Find acyclic energy-feasible paths in the network from source src to
    # destination dest with/without excluding nodes with self loop
    # args: source (src), destination (dest), energy budget (EB)
    # TODO: Finding paths with cycles in the network
    # This function can find only those paths in which a node is visited just once
    def findPaths(self, src, dest, EB: number=infinity, excludeSelfLoopNodes:
            bool=False, verbose: bool=False) -> List[Path]:
        if excludeSelfLoopNodes:
            # Find nodes with self loops
            selfLoopNodes = [e.node_from for e in self.edges if e.node_from == e.node_to]
        else:
            selfLoopNodes = []
        # print('Nodes with self loop: ', *(n for n in selfLoopNodes))

        # Queue to store (partial) paths
        q = deque()

        # Add edges going out of the source to q
        for e in self.edges:
            if e.node_from == src:
                q.append(Path([e]))

        # List to store the final paths
        pathList = []
        count = 0

        while q:
            count += 1
            if verbose: print("\ncount:%d"%count)

            # Get the (earliest generated) partial path
            if verbose: print("q before pop")
            for p in q:
                if verbose: print(printPathInNetwork(p, self))
            path = q.popleft()
            # print("after pop ", printPathInNetwork(path, self), q)
            for p in q:
                if verbose: printPathInNetwork(p, self)

            # Get the last node in the partial path
            last = path.edges[-1].node_to
            if verbose: print("last ", last)

            # If the last node is the destination node then store the path
            if last == dest:
                if verbose: print("Found s-t Path:", printPathInNetwork(path, self))
                pathList.append(path)

            # Traverse all the nodes connected to the current node and push new partial
            # path to queue
            edgeListCurrNode = [e for e in self.edges if (e.node_from == last and
                e.node_to not in selfLoopNodes)]
            if verbose: print("edgeListCurrNode ", len(edgeListCurrNode), edgeListCurrNode)
            for e in edgeListCurrNode:
                if verbose: print("edge %d" %self.edges.index(e), e,
                        printPathInNetwork(path, self), path.isNodeInPath(e.node_to),
                        path.getEnergyConsump(), e.ec, (path.getEnergyConsump() + e.ec <= EB))
                if (not path.isNodeInPath(e.node_to)) and (path.getEnergyConsump() + e.ec <= EB):
                    newpath = Path(path.edges)
                    if verbose: print("newpath before append ", printPathInNetwork(newpath,self))
                    newpath.add_edge_at_end(e)
                    q.append(newpath)
                    if verbose: print("newpath after append ", printPathInNetwork(newpath, self))

        # Print pathList
        if verbose:
            print("\nTotal %d paths found from node %s to node %s:"%(len(pathList),src,dest))
            for i,p in enumerate(pathList):
                print(i, len(p), printPathInNetwork(p, self))
        return pathList

    # Find (cyclic) energy-feasible paths in the network from source src to
    # destination dest including ones generated using concatenation of subpaths (i)
    # from src to nodes with self-loop (ii) from nodes with self-loop to dest (iii)
    # between nodes with self-loop
    # args: source (src), destination (dest), energy budget (EB)
    def findPathsWithLoops(self, src, dest, EB: number=infinity, verbose: bool=False) -> List[Path]:
        # (i) All energy feasible paths from src to dest
        pathList = self.findPaths(src, dest, EB, excludeSelfLoopNodes=True)
        # print('paths: ',*(printPathInNetwork(p, self) for p in pathList), sep='\n')

        # (ii) All energy feasible paths involving nodes with self loops
        # Each such path will contain a node with self loop atmost once
        # Find nodes with self loops
        selfLoopNodes = [e.node_from for e in self.edges if e.node_from == e.node_to]
        pathCombs = list(combo for r in range(len(selfLoopNodes)) for combo in
                itertools.combinations(selfLoopNodes, r+1))
        allCombs = []
        for c in pathCombs:
            allCombs = [*allCombs, *list(itertools.permutations(c))]
        # for c in allCombs:
            # print('c: ', c)
            # print('i', *(i for i in (c)))
        # print('combs: ',*(list(c) for c in allCombs), sep='\n')
        # for c in allCombs:
            # print('combs:',*(i for i in list(c)), sep='\n')
            # print('combs:',list(c), sep='\n')
        for comb in allCombs:
            print('comb: ',*(i for i in list(comb)), sep=',')
            pathsComb = self.findPaths(src, comb[0], EB)
            k = 1
            while k < len(comb):
                pathsComb = self.joinFeasiblePaths(pathsComb,
                        self.findPaths(comb[k-1], comb[k], EB), EB)
                k += 1
            pathList = [*pathList, *self.joinFeasiblePaths(pathsComb,\
                    self.findPaths(comb[-1], dest, EB), EB)]
            # print('pathsComb: ', *((self.printPathInNetwork(p),p.getEnergyConsump())
                # for p in self.joinFeasiblePaths(pathsComb, self.findPaths(comb[-1],
                    # dest, EB), EB)), sep='\n')
        print('pathsList: ', *((self.printPathInNetwork(p),p.getNetEnergyConsump(),\
                p.getFreeFlowTravelTime()) for p in pathList), sep='\n')
        # exit(0)
        return pathList


# A directed path
class Path:
    edges: List[Edge]
    firstNode: Node

    def __init__(self, edges: List[Edge], start: Node=None):
        # For the empty path we explicitely have to specify the start node
        assert(len(edges) > 0 or start is not None)
        if start is None:
            start = edges[0].node_from
        self.firstNode = start

        currentNode = start
        self.edges = []
        for e in edges:
            # Check whether edges do form a path:
            assert(currentNode == e.node_from)
            self.edges.append(e)
            currentNode = e.node_to

    def add_edge_at_start(self,e : Edge):
        # Adds an edge at the beginning of a path
        assert(e.node_to == self.getStart())
        self.edges.insert(0,e)
        self.firstNode = e.node_from

    def add_edge_at_end(self,e : Edge):
        # Adds an edge at the end of a path
        assert (e.node_from == self.getEnd())
        self.edges.append(e)

    def getStart(self) -> Node:
        return self.firstNode

    def getEnd(self) -> Node:
        if len(self.edges) == 0:
            return self.firstNode
        else:
            return self.edges[-1].node_to

    def getFreeFlowTravelTime(self):
        # TODO: put in checks
        fftt = 0
        for e in self.edges:
            fftt += e.tau
        return fftt

    def getEnergyConsump(self):
        # TODO: put in checks
        ec = 0
        for e in self.edges:
            ec += e.ec
        return ec

    def getNetEnergyConsump(self):
        # TODO: put in checks
        ec = 0
        for e in self.edges:
            if not (e.node_from == e.node_to): ec += abs(e.ec)
        return ec

    def getNodesInPath(self) -> List[Node]:
        nodeList = [self.firstNode]
        for e in self.edges:
            nodeList.append(e.node_to)
        return nodeList

    def __len__(self):
        return len(self.edges)

    def __str__(self) -> str:
        s = str(self.firstNode) + "-" + str(self.getEnd()) + " path: "
        for e in self.edges:
            s += str(e)
        return s

    def __eq__(self, other : Path) -> bool:
        if not isinstance(other,Path):
            return False
        if len(self.edges) != len(other.edges):
            return False
        for i in range(len(self.edges)):
            if self.edges[i] != other.edges[i]:
                return False
        return True

    def __hash__(self):
        # TODO: This is a very bad way of hashing
        # If possible, this should be avoided at all as Path-objects are not immutable
        h = hash(self.firstNode)
        for e in self.edges:
            h = 2*h + hash(e)
        return h

    # Function to check if a node is part of the path
    def isNodeInPath(self, x: Node) -> bool:
        # print("nodes in path ", str(self.getNodesInPath()))
        if x in self.getNodesInPath():
            # print("False")
            return True
        else:
            # print("True")
            return False


# Creates a random series parallel network with m edges
# The source node will be named s, the sink node t. All other nodes are nameless
def createRandomSPnetwork(m:int) -> Network:
    # TODO: Also create random capacities and travel times
    # In some structured way? Maybe let the user specify some point between the following two extremes:
    # - completely deterministic: Subdividing an edge subdivides the lengths, duplicating splits the capacity
    # - completely random: All edges get random capacities and travel times
    assert(m>0)

    network = Network()
    network.addNode("s")
    network.addNode("t")
    network.addEdge("s","t",1,1)
    m -= 1

    random.seed()
    for _ in range(m):
        e = random.choice(network.edges)
        if random.choice([True,False]):
            network.subdivide(e,1,1,1)
        else:
            network.duplicate(e,1,1,1)
    return network

# Print path p based on edge ids in the network G
def printPathInNetwork(p: Path, G: Network):
    s = str()
    for e in p.edges:
        if len([i for i in e.node_from.outgoing_edges\
                if i.node_to == e.node_to]) > 1:
            s += str(G.edges.index(e))
        s += str(e)
    return s

# TODO: a function to get all energy feasbile s-t paths in a network N, given
# nodes s and t
def getPathList(self, network: Network, sourceNode: Node, sinkNode: Node):
    print("To be implemented")


