from __future__ import annotations

import heapq, math
from typing import List, Dict

from network import *
from flows import *
from utilities import *

def networkLoading(network: Network, pathFlows: List[(Path,PWConst)], timeHorizon: ExtendedRational=math.inf) -> PartialFlow:
    # Determining a (partial) flow from a set of given path inflow rates
    noOfCommodities = len(pathFlows)
    partialPathFlows = [PartialPathFlow(path) for (path,_) in pathFlows]
    flow = PartialFlow(network,noOfCommodities)

    class Event:
        time: ExtendedRational
        v: Node

        def __init__(self,time: ExtendedRational,v: Node):
            self.time = time
            self.v = v

        def __lt__(self, other):
            return self.time < other.time

        def __str__(self):
            return "Event at node "+str(self.v) + " at time " + str(float(self.time)) + " ≈ " + str(self.time)

    eventQueue: List[Event] = []

    flowTerminated = False # TODO: Das als Abbruchbedingung einbauen

    for v in network.nodes:
        heapq.heappush(eventQueue,Event(ExtendedRational(0),v))

    while len(eventQueue) > 0:

        event = heapq.heappop(eventQueue)
        print("Handling ", event)
        v = event.v
        theta = event.time
        if theta >= timeHorizon:
            return flow
        if flow.upToAt[v] == theta:
            # TODO: Nur Tatsaechliche Events betrachten

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
                # if pathFlows[i][0].firstNode == v:
                #     partialPathFlows[i].fPlus[0].addSegment(nextTheta,pathFlows[i][1].getValueAt(theta))
                #     e = partialPathFlows[i].path.edges[0]
                #     if flowTo[e] > 0:
                #         outflowRate = pathFlows[i][1].getValueAt(theta)/flowTo[e] * min(flowTo[e],e.nu)
                #     else:
                #         outflowRate = 0
                #     partialPathFlows[i].fMinus[0].addSegment(flow.T(e,nextTheta),outflowRate)

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
            heapq.heappush(eventQueue,Event(nextTheta,v))

    return flow
