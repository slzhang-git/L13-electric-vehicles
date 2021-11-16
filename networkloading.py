from __future__ import annotations

import heapq, math
from typing import List, Dict

from network import *
from flows import *
from utilities import *


class Event:
    # A class for events in the network loading algorithm
    # An event consists of a time and a node
    # Such an event denotes a time at which a new flow split has to be calculated at the given node
    # (because the amount of incoming flow into the node potentially changes after this time)
    time: ExtendedRational
    v: Node

    def __init__(self, time: ExtendedRational, v: Node):
        self.time = time
        self.v = v

    def __lt__(self, other):
        return self.time < other.time

    def __str__(self):
        return "Event at node " + str(self.v) + " at time " + str(float(self.time)) + " ≈ " + str(self.time)

class EventQueue:
    # A queue of events, where events are ordered by non-decreasing trigger time
    events: List[Event]

    def __init__(self):
        self.events = []

    # Adds a new event to the queue
    def pushEvent(self,time:ExtendedRational,v:Node):
        heapq.heappush(self.events, Event(time, v))

    # Returns the next event and removes it from the queue
    def popEvent(self) -> Event:
        return heapq.heappop(self.events)

    # Indicates whether there are any events left in the queue
    def isEmpty(self) -> bool:
        return len(self.events) == 0


# Determines the edge based description of a flow given the path inflow rates
# The arguments are:
# - pathBasedFlows: An object of class PartialFlowPathBased containing a set of commodities and for every commodity
#   its source, sink, network inflow rate and a list of tuples (path p, inflow rate into path p)
# - timeHorizon (optional): the flow will be determined for the time interval [0,timeHorizon]; default is infinity
# - verbose: If true more information is printed during the network loading procedure; default is False
def networkLoading(pathBasedFlows : PartialFlowPathBased, timeHorizon: ExtendedRational=math.inf, verbose:bool=False) -> PartialFlow:
    network = pathBasedFlows.network

    # We start by creating a list of partial path flows.
    # For every commodity and each of its used paths, the corresponding path flow will contain edge in- and outflow rates
    # for every edge this path
    # Note, that edges may occur multiple times on a path and then also have multiple in-/outflow functions here
    # This is important to keep track on where particles leaving an edge are supposed to travel next
    partialPathFlows = []
    for i in range(pathBasedFlows.getNoOfCommodities()):
        partialPathFlows.extend([PartialPathFlow(path) for path in pathBasedFlows.fPlus[i]])

    # During the network loading we consider every path of each commodity as its own commodity:
    # TODO: maybe convert back into a flow with only one in-/outflow function for every true commodity at the end?
    noOfCommodities = len(partialPathFlows)

    # This will be the actual flow (i.e. edge in-/outflow rates for all edges and commodities and queues for all edges
    flow = PartialFlow(network,noOfCommodities)

    # For all commodities (corresponding to one source-sink path each) set source, sink and network inflow rate
    j = 0
    for i in range(pathBasedFlows.getNoOfCommodities()):
        for p in pathBasedFlows.fPlus[i]:
            flow.setSource(j, pathBasedFlows.sources[i])
            flow.setSink(j, pathBasedFlows.sinks[i])
            flow.setU(j, pathBasedFlows.fPlus[i][p])
            j += 1

    # A queue of node events:
    eventQueue = EventQueue()
    # During the network loading the following invariant always holds (at each node v separately):
    # There is some time \theta such that the edge inflow rates for all edges leaving v are defined up to time \theta
    # and if \theta < timeHorizon there is an event (\theta,v) in the queue (sigifying that at that time a new flow
    # distribution has to be determined at node v
    for v in network.nodes:
        eventQueue.pushEvent(ExtendedRational(0),v)

    flowTerminated = False  # TODO: Das als Abbruchbedingung einbauen

    # While there are events left to handle
    while not eventQueue.isEmpty():
        event = eventQueue.popEvent()
        if verbose: print("Handling ", event)
        v = event.v
        theta = event.time
        # currently all edge inflow rates for edges leaving v are defined up to time \theta
        assert(flow.upToAt[v] == theta)
        # We will now extend them for some additional time interval [theta,nextTheta]

        # At most, we extend until the given timeHorizon
        nextTheta = timeHorizon

        # For each edge leaving v we will add up (over all commodities) the total volume of flow we have to send over this edge
        flowTo = {e: 0 for e in v.outgoing_edges}
        for i in range(noOfCommodities):
            if flow.sources[i] == v:
                # If node v is the source of the current commodity i then this commodities network inflow rate u
                # determines the amount of flow we have to send.
                nextTheta = min(nextTheta,flow.u[i].getNextStepFrom(theta))
                flowTo[partialPathFlows[i].path.edges[0]] += flow.u[i].getValueAt(theta)

            for j in range(len(partialPathFlows[i].path.edges)-1):
                # Stepping through the current commodities path to check whether node v occurs on it as an edges endnode
                # (last edge can be ignored as flow leaving this edges leaves the network)
                if partialPathFlows[i].path.edges[j].node_to == v:
                    # If so, the outflow from this edge will be sent to the next edge on the path
                    nextTheta = min(nextTheta,partialPathFlows[i].fMinus[j].getNextStepFrom(theta))
                    flowTo[partialPathFlows[i].path.edges[j+1]] += partialPathFlows[i].fMinus[j].getValueAt(theta)

        # Check whether queues on outgoing edges deplete before nextTheta
        # (and reduce nextTheta in that case)
        for e in v.outgoing_edges:
            # The queue length at time theta:
            qeAtTheta = flow.queues[e].getValueAt(theta)
            if qeAtTheta > 0 and flowTo[e] < e.nu:
                # the time at which the queue on e will run empty (if the inflow into e remains constant)
                x = qeAtTheta/(e.nu-flowTo[e])
                # If this is before the current value of nextTheta we reduce nextTheta to the time the queue runs empty
                nextTheta = min(nextTheta,theta+x)

        # Now we have all the information to extend the current flow at node v over the time interval [theta,nextTheta]:

        # Determine queues on the interval [theta,nextTheta]
        for e in v.outgoing_edges:
            if flow.queues[e].getValueAt(theta) > 0:
                flow.queues[e].addSegmant(nextTheta,flowTo[e]-e.nu)
            else:
                flow.queues[e].addSegmant(nextTheta, max(flowTo[e] - e.nu,0))

        # Redistribute the node inflow to outgoing edges
        for i in range(noOfCommodities):
            # First we determine the values of f^+/f^- in terms of the path flows
            for j in range(len(partialPathFlows[i].path.edges)):
                if partialPathFlows[i].path.edges[j].node_from == v:
                    e = partialPathFlows[i].path.edges[j]
                    # Determine inflow into edge e
                    if j == 0:
                        # The edge is the first on commodity i's path.
                        # Thus the inflow into this edge is determined by the network inflow rate u_i
                        inflow = flow.u[i].getValueAt(theta)
                    else:
                        # Otherwise the inflow is determined by the outflow of the previous edge on i's path
                        inflow = partialPathFlows[i].fMinus[j-1].getValueAt(theta)
                    # Adjust fPlus of edge e
                    partialPathFlows[i].fPlus[j].addSegment(nextTheta,inflow)
                    # Adjust fMinus of edge e
                    if flowTo[e] > 0:
                        outflowRate = partialPathFlows[i].fPlus[j].getValueAt(theta) / flowTo[e] * min(flowTo[e], e.nu)
                    else:
                        outflowRate = 0
                    if nextTheta < ExtendedRational(1,0):
                        partialPathFlows[i].fMinus[j].addSegment(flow.T(e, nextTheta), outflowRate)
                    else:
                        partialPathFlows[i].fMinus[j].addSegment(ExtendedRational(1,0), outflowRate)

            # Now we convert the path flows into the actual edge in- and outflow rates
            # i.e. if an edge occurs multiple times on a commodity's path we add up the corresponding rates from the path flow
            for e in v.outgoing_edges:
                inflowRate = ExtendedRational(0)
                outflowRate = ExtendedRational(0)
                for j in range(len(partialPathFlows[i].path.edges)):
                    if partialPathFlows[i].path.edges[j] == e:
                        inflowRate += partialPathFlows[i].fPlus[j].getValueAt(theta)
                        outflowRate += partialPathFlows[i].fMinus[j].getValueAt(flow.T(e,theta))
                flow.fPlus[(e,i)].addSegment(nextTheta,inflowRate)
                if nextTheta < ExtendedRational(1,0):
                    flow.fMinus[(e, i)].addSegment(flow.T(e,nextTheta), outflowRate)
                else:
                    flow.fMinus[(e, i)].addSegment(ExtendedRational(1,0), outflowRate)

        # Now the extension at node v is done -> we have a flow up to time nextTheta at node v
        flow.upToAt[v] = nextTheta

        # At time nextTheta we will have to determine a new flow distribution at node v
        # (unless we already reached our desired time horizon then)
        if nextTheta < timeHorizon:
            eventQueue.pushEvent(nextTheta,v)

    return flow
