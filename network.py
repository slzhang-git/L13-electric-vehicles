from __future__ import annotations

from utilities import *

# A directed edge (from https://github.com/Schedulaar/predicted-dynamic-flows/blob/main/predictor/src/core/graph.py )
# with capacity nu and travel time tau
class Edge:
    node_from: Node
    node_to: Node
    tau: ExtendedRational
    nu: ExtendedRational

    def __init__(self, node_from: Node, node_to: Node, capacity: ExtendedRational=1, traveltime: ExtendedRational=1):
        # Creating an edge from node_from to node_to
        self.node_from = node_from
        self.node_to = node_to

        assert(traveltime >= 0)
        self.tau = traveltime

        assert(capacity >= 0)
        self.nu = capacity


    def __str__(self):
        return "("+str(self.node_from)+","+str(self.node_to)+")"

# A node (from https://github.com/Schedulaar/predicted-dynamic-flows/blob/main/predictor/src/core/graph.py )
class Node:
    name: str
    incoming_edges: List[Edge]
    outgoing_edges: List[Edge]

    def __init__(self, name: str):
        # Create a node with name name and without incoming or outgoing edges
        self.name = name
        self.incoming_edges = []
        self.outgoing_edges = []

    def __str__(self):
        # Print the nodes name
        return str(self.name)

# A network consisting of a directed graph with capacities and travel times on all edges
class Network:
    edges: List[Edge]
    nodes: List[Node]

    def __init__(self):
        # Create an empty network
        self.edges = []
        self.nodes = []

    def getNode(self,name: str):
        for v in self.nodes:
            if v.name == name:
                return v
        return None

    def addNode(self,name: str):
        # TODO: Unique Node-Names?
        self.nodes.append(Node(name))

    def addEdge0(self,node_from: Node, node_to: Node, nu: ExtendedRational, tau: ExtendedRational):
        assert (node_from in self.nodes and node_to in self.nodes)
        e = Edge(node_from, node_to, nu, tau)
        node_from.outgoing_edges.append(e)
        node_to.incoming_edges.append(e)
        self.edges.append(e)

    def addEdge(self,node_from: str, node_to: str, nu: ExtendedRational, tau: ExtendedRational):
        v = self.getNode(node_from)
        w = self.getNode(node_to)
        self.addEdge0(v,w,nu,tau)


# A directed path
class Path:
    edges: List[Edge]
    firstNode: Node

    def __init__(self, edges: List[Edge]):
        self.edges = []
        for e in edges:
            if len(self.edges) > 0:
                # Check whether edges do form a path:
                assert(self.edges[-1].node_to == e.node_from)
            self.edges.append(e)
        # TODO: Was machen bei leerem Weg?
        assert(len(self.edges)>0)
        self.firstNode = self.edges[0].node_from

    def __len__(self):
        return len(self.edges)