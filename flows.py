from __future__ import annotations

from typing import Dict, List
from network import *
from utilities import *

# A partial feasible flow with in-/outflow rates for every edges and commodity and queues for every edge
# Feasibility is not checked automatically but a check can be initiated by calling .checkFeasibility
class PartialFlow:
    network: Network
    upToAt: Dict[Node, ExtendedRational]
    fPlus: Dict[(Edge,int), PWConst]
    fMinus: Dict[(Edge,int), PWConst]
    queues: Dict[Edge, PWLin]
    sources: List[Node]
    sinks: List[Node]
    u: List[PWConst]
    noOfCommodities: int

    # TODO: Add a flag for terminating flow?

    def __init__(self, network: Network, numberOfCommodities: int):
        self.network = network
        self.noOfCommodities = numberOfCommodities

        # The zero-flow up to time zero
        self.upToAt = {}
        for v in network.nodes:
            self.upToAt[v] = ExtendedRational(0)

        # Initialize functions f^+,f^- and q for every edge e
        self.fPlus = {}
        self.fMinus = {}
        self.queues = {}
        for e in network.edges:
            self.queues[e] = PWLin([ExtendedRational(0), ExtendedRational(0)], [ExtendedRational(0)], [ExtendedRational(0)])
            for i in range(numberOfCommodities):
                self.fPlus[(e,i)] = PWConst([ExtendedRational(0),ExtendedRational(0)],[ExtendedRational(0)])
                self.fMinus[(e,i)] = PWConst([ExtendedRational(0),e.tau],[ExtendedRational(0)])

        # Currently every commodity has a network inflow rate of zero
        self.u = [PWConst([ExtendedRational(0)],[],0) for _ in range(self.noOfCommodities)]
        # Furthermore we do not know source and sink nodes yet
        self.sources = [None for _ in range(self.noOfCommodities)]
        self.sinks = [None for _ in range(self.noOfCommodities)]

    def setSource(self,commodity:int,s:Node):
        assert(commodity < self.noOfCommodities)
        self.sources[commodity] = s

    def setSink(self,commodity:int,t:Node):
        assert(commodity < self.noOfCommodities)
        self.sinks[commodity] = t

    def setU(self,commodity:int,u:PWConst):
        assert (commodity < self.noOfCommodities)
        self.u[commodity] = u



    # The travel time over an edge e at time theta
    def c(self, e:Edge, theta:ExtendedRational) -> ExtendedRational:
        # the queue on edge e has to be known up to at least time theta
        assert (self.queues[e].segmentBorders[-1] >= theta)
        return self.queues[e].getValueAt(theta)/e.nu + e.tau

    # The arrival time at the end of edge e if entering at time theta
    def T(self, e:Edge, theta:ExtendedRational) -> ExtendedRational:
        return theta + self.c(e, theta)

    # Determines the arrival time at the end of path p when starting at time theta
    def pathArrivalTime(self,p:Path,theta:ExtendedRational)->ExtendedRational:
        # TODO: check whether all necessary queues are available
        if len(p) == 0:
            return theta
        else:
            firstEdge = p.edges[0]
            return self.pathArrivalTime(Path(p.edges[1:],firstEdge.node_to),self.T(firstEdge,theta))



    def checkFlowConservation(self,v: Node,upTo: ExtendedRational,commodity: int) -> bool:
        # Checks whether flow conservation holds at node v during the interval [0,upTo]
        # i.e. \sum_{e \in \delta^-(v)}f_e,i^-(\theta) = \sum_{e \in \delta^+(v)}f_e,i^+(\theta)
        # for all nodes except source and sink of commodity i
        # For the source the same has to hold with the network inflow rate u of commodity i added on the left side
        # For the sink the = is replaced by >=
        theta = ExtendedRational(0)
        # Since all flow rates are assumed to be right-constant, it suffices to check the conditions
        # at every stepping point (for at least one of the relevant flow rate functions)
        while theta < upTo:
            nextTheta = ExtendedRational(1, 0)
            flow = ExtendedRational(0)
            # Add up node inflow rate (over all incoming edges)
            for e in v.incoming_edges:
                flow += self.fMinus[e,commodity].getValueAt(theta)
                nextTheta = min(nextTheta,self.fMinus[e,commodity].getNextStepFrom(theta))
            # If node v is commodity i's source we also add the network inflow rate
            if v == self.sources[commodity]:
                flow += self.u[commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self.u[commodity].getNextStepFrom(theta))
            # Subtract all node outflow (over all outgoing edges)
            for e in v.outgoing_edges:
                flow -= self.fPlus[e, commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self.fPlus[e, commodity].getNextStepFrom(theta))

            # Now check flow conservation at node v
            # First, a special case for the sink node
            if v == self.sinks[commodity]:
                if flow < 0:
                    # TODO: Fehlermeldung
                    print("Flow conservation does not hold at node ", v, " (sink!) at time ", theta)
                    return False

            # Then the case for all other nodes:
            elif flow != 0:
                # TODO: Fehlermeldung
                print("Flow conservation does not hold at node ",v," at time ",theta)
                return False

            # The next stepping point:
            theta = nextTheta
        return True

    # Checks whether queues operate at capacity, i.e. whether the edge outflow rate is determined by
    # f^-_e(\theta+\tau_e) = \nu_e,                     if q_e(\theta) > 0
    # f^-_e(\theta+\tau_e) = \min{\nu_e,f^+_e(\theta)}, else
    def checkQueueAtCap(self, e: Edge, upTo: ExtendedRational) -> bool:
        theta = ExtendedRational(0)

        while theta < upTo:
            nextTheta = ExtendedRational(1, 0)
            outflow = ExtendedRational(0)
            inflow = ExtendedRational(0)
            for i in range(self.noOfCommodities):
                outflow += self.fMinus[(e,i)].getValueAt(theta+e.tau)
                inflow += self.fPlus[(e,i)].getValueAt(theta)
                nextTheta = min(nextTheta,self.fMinus[(e,i)].getNextStepFrom(theta+e.tau),self.fPlus[(e,i)].getNextStepFrom(theta))
            if self.queues[e].getValueAt(theta) > 0:
                if outflow != e.nu:
                    # TODO: Fehlermeldung
                    print("Queue on edge ",e, " does not operate at capacity at time ", theta)
                    return False
            else:
                assert(self.queues[e].getValueAt(theta) == 0)
                if outflow != min(inflow,e.nu):
                    # TODO: Fehlermeldung
                    print("Queue on edge ", e, " does not operate at capacity at time ", theta)
                    return False
            theta = nextTheta
        return True

    # Checks whether the queue-lengths are correct, i.e. determined by
    # q_e(\theta) = F_e^+(\theta) - F_e^-(\theta+\tau_e)
    def checkQueue(self,e: Edge,upTo: ExtendedRational):
        # Assumes that f^-_e = 0 on [0,tau_e)
        theta = ExtendedRational(0)
        currentQueue = ExtendedRational(0)
        if self.queues[e].getValueAt(theta) != 0:
            # TODO: Fehlermeldung
            print("Queue on edge ", e, " does not start at 0")
            return False
        while theta < upTo:
            nextTheta = self.queues[e].getNextStepFrom(theta)
            inflow = ExtendedRational(0)
            outflow = ExtendedRational(0)
            for i in range(self.noOfCommodities):
                outflow += self.fMinus[(e, i)].getValueAt(theta + e.tau)
                inflow += self.fPlus[(e, i)].getValueAt(theta)
                nextTheta = min(nextTheta,self.fPlus[(e,i)].getNextStepFrom(theta),self.fMinus[(e,i)].getNextStepFrom(theta+e.tau))
            currentQueue += (inflow-outflow)*(nextTheta-theta)
            if currentQueue != self.queues[e].getValueAt(nextTheta):
                # TODO: Fehlermeldung
                print("Queue on edge ", e, " wrong at time ", theta)
                return False
            theta = nextTheta
        return True

    # Check feasibility of the given flow up to the specified time horizon
    def checkFeasibility(self,upTo: ExtendedRational) -> bool:
        # Does not check FIFO (TODO)
        # Does not check non-negativity (TODO?)
        feasible = True
        for i in range(self.noOfCommodities):
            for v in self.network.nodes:
                feasible = feasible and self.checkFlowConservation(v,upTo,i)
        for e in self.network.edges:
            feasible = feasible and self.checkQueueAtCap(e,upTo)
            feasible = feasible and self.checkQueue(e,upTo)
        return feasible

    # Returns a string representing the queue length functions as well as the edge in and outflow rates
    def __str__(self):
        s = "Queues:\n"
        for e in self.network.edges:
            s += " q_" + str(e) + ": " + str(self.queues[e]) + "\n"
        s += "----------------------------------------------------------\n"
        for i in range(self.noOfCommodities):
            s += "Commodity " + str(i) + ":\n"
            for e in self.network.edges:
                s += "f_" + str(e) + ": " + str(self.fPlus[(e,i)]) + "\n"
                s += "f_" + str(e) + ": " + str(self.fMinus[(e,i)]) + "\n"
            s += "----------------------------------------------------------\n"
        return s


# A path based description of feasible flows:
class PartialFlowPathBased:
    network: Network
    upToFor : List[ExtendedRational]
    fPlus: List[Dict[Path, PWConst]]
    sources: List[Node]
    sinks: List[Node]
    noOfCommodities: int

    def __init__(self, network: Network, numberOfCommodities: int):
        self.network = network
        self.noOfCommodities = numberOfCommodities

        # The zero-flow up to time zero
        self.upToFor = [ExtendedRational(0) for _ in range(numberOfCommodities)]

        # Initialize functions f^+
        self.fPlus = [{} for _ in range(numberOfCommodities)]

        # Currently every commodity has a network inflow rate of zero
        self.u = [PWConst([ExtendedRational(0)],[],0) for _ in range(self.noOfCommodities)]
        # Furthermore we do not know source and sink nodes yet
        self.sources = [None for _ in range(self.noOfCommodities)]
        self.sinks = [None for _ in range(self.noOfCommodities)]

    def setPaths(self, commodity:int, paths:List[Path], pathinflows:List[PWConst]):
        assert (commodity < self.noOfCommodities)
        assert (len(paths) > 0)
        self.sources[commodity] = paths[0].getStart()
        self.sinks[commodity] = paths[0].getEnd()

        assert (len(paths) == len(pathinflows))
        for i in range(len(paths)):
            p = paths[i]
            assert (p.getStart() == self.sources[commodity])
            assert (p.getEnd() == self.sinks[commodity])
            fp = pathinflows[i]
            assert (not p in self.fPlus[commodity])
            self.fPlus[commodity][p] = fp


    def getNoOfCommodities(self) -> int:
        return self.noOfCommodities



class PartialPathFlow:
    path: Path
    fPlus: List[PWConst]
    fMinus: List[PWConst]

    def __init__(self,path):
        self.path = path
        self.fPlus = []
        self.fMinus = []
        for e in path.edges:
            self.fPlus.append(PWConst([ExtendedRational(0), ExtendedRational(0)], [ExtendedRational(0)], ExtendedRational(0)))
            self.fMinus.append(PWConst([ExtendedRational(0), e.tau], [ExtendedRational(0)], ExtendedRational(0)))

