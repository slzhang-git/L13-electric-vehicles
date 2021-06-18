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

def networkLoading(network: Network, pathFlows: List[(Path,PWConst)], timeHorizon: ExtendedRational=math.inf, verbose:bool=False) -> PartialFlow:
    # Determining a (partial) flow from a set of given path inflow rates
    noOfCommodities = len(pathFlows)
    # This is a helper object containing edge in and outflow rates for every edge
    # on the commodity-specific path
    # Note that edges may occure multiple times on a path and then also have multiple in-/outflow functions here
    # This is important to keep track on where particles leaving an edge are supposed to travel next
    partialPathFlows = [PartialPathFlow(path) for (path,_) in pathFlows]
    # This is the actual flow (i.e. edge in-/outflow rates for all edges and commodities and queues for all edges
    flow = PartialFlow(network,noOfCommodities)

    # For all commodities (corresponding to one source-sink path each) set source, sink and network inflow rate
    for i in range(noOfCommodities):
        flow.setSource(i, pathFlows[i][0].getStart())
        flow.setSink(i, pathFlows[i][0].getEnd())
        flow.setU(i, pathFlows[i][1])

    # A queue of node events
    eventQueue = EventQueue()
    for v in network.nodes:
        eventQueue.pushEvent(ExtendedRational(0),v)

    flowTerminated = False  # TODO: Das als Abbruchbedingung einbauen

    while not eventQueue.isEmpty():
        event = eventQueue.popEvent()
        if verbose: print("Handling ", event)
        v = event.v
        theta = event.time
        if theta >= timeHorizon:
            if verbose:
                print("Remaining unhandled events:")
                while not eventQueue.isEmpty():
                    print(eventQueue.popEvent())
            return flow

        assert(flow.upToAt[v] == theta)

        nextTheta = timeHorizon
        flowTo = {e: 0 for e in v.outgoing_edges}

        for i in range(noOfCommodities):
            if pathFlows[i][0].firstNode == v:
                nextTheta = min(nextTheta,pathFlows[i][1].getNextStepFrom(theta))
                flowTo[pathFlows[i][0].edges[0]] += pathFlows[i][1].getValueAt(theta)

            for j in range(len(pathFlows[i][0].edges)-1):
                # Last edge can be ignored as flow leaving this edges leaves the network
                if pathFlows[i][0].edges[j].node_to == v:
                    nextTheta = min(nextTheta,partialPathFlows[i].fMinus[j].getNextStepFrom(theta))
                    flowTo[pathFlows[i][0].edges[j+1]] += partialPathFlows[i].fMinus[j].getValueAt(theta)

        # Check whether queues on outgoing edges deplete before nextTheta
        # (and reduce nextTheta in that case)
        for e in v.outgoing_edges:
            qeAtTheta = flow.queues[e].getValueAt(theta)
            if qeAtTheta > 0 and flowTo[e] < e.nu:
                # the time at which the queue on e will run empty (if the inflow into e remains constant)
                x = qeAtTheta/(e.nu-flowTo[e])
                nextTheta = min(nextTheta,theta+x)

        # Determine queues on the interval [theta,nextTheta]
        for e in v.outgoing_edges:
            if flow.queues[e].getValueAt(theta) > 0:
                flow.queues[e].addSegmant(nextTheta,flowTo[e]-e.nu)
            else:
                flow.queues[e].addSegmant(nextTheta, max(flowTo[e] - e.nu,0))

        # Redistribute the node inflow to outgoing edges
        for i in range(noOfCommodities):
            for j in range(len(pathFlows[i][0].edges)):
                if pathFlows[i][0].edges[j].node_from == v:
                    e = partialPathFlows[i].path.edges[j]
                    # Determine inflow into edge e
                    if j == 0:
                        inflow = pathFlows[i][1].getValueAt(theta)
                    else:
                        inflow = partialPathFlows[i].fMinus[j-1].getValueAt(theta)
                    # Adjust fPlus of edge e
                    partialPathFlows[i].fPlus[j].addSegment(nextTheta,inflow)
                    # Adjust fMinus of edge e
                    if flowTo[e] > 0:
                        outflowRate = partialPathFlows[i].fPlus[j].getValueAt(theta) / flowTo[e] * min(flowTo[e], e.nu)
                    else:
                        outflowRate = 0
                    partialPathFlows[i].fMinus[j].addSegment(flow.T(e, nextTheta), outflowRate)

            for e in v.outgoing_edges:
                inflowRate = ExtendedRational(0)
                outflowRate = ExtendedRational(0)
                for j in range(len(pathFlows[i][0])):
                    if pathFlows[i][0].edges[j] == e:
                        inflowRate += partialPathFlows[i].fPlus[j].getValueAt(theta)
                        outflowRate += partialPathFlows[i].fMinus[j].getValueAt(flow.T(e,theta))
                flow.fPlus[(e,i)].addSegment(nextTheta,inflowRate)
                flow.fMinus[(e, i)].addSegment(flow.T(e,nextTheta), outflowRate)

        flow.upToAt[v] = nextTheta
        eventQueue.pushEvent(nextTheta,v)

    return flow
