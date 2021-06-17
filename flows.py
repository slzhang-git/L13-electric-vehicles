from __future__ import annotations

from network import *
from utilities import *

# A partial feasible flow
class PartialFlow:
    network: Network
    upToAt: Dict[Node, ExtendedRational]
    fPlus: Dict[(Edge,int), PWConst]
    fMinus: Dict[(Edge,int), PWConst]
    queues: Dict[Edge, PWLin]
    noOfCommodities: int

    def __init__(self, network: Network, numberOfCommodities: int):
        self.network = network
        self.noOfCommodities = numberOfCommodities
        # The zero-flow up to time zero
        self.upToAt = {}
        for v in network.nodes:
            self.upToAt[v] = ExtendedRational(0)

        self.fPlus = {}
        self.fMinus = {}
        self.queues = {}

        for e in network.edges:
            self.queues[e] = PWLin([ExtendedRational(0), ExtendedRational(0)], [ExtendedRational(0)], [ExtendedRational(0)])
            for i in range(numberOfCommodities):
                self.fPlus[(e,i)] = PWConst([ExtendedRational(0),ExtendedRational(0)],[ExtendedRational(0)])
                self.fMinus[(e,i)] = PWConst([ExtendedRational(0),e.tau],[ExtendedRational(0)])

    def c(self, e:Edge, theta:ExtendedRational) -> ExtendedRational:
        # the queue on edge e has to be known up to at least time theta
        assert (self.queues[e].segmentBorders[-1] >= theta)
        #print("Waiting time on ", e, "at ", theta, " is ", self.queues[e].getValueAt(theta)/e.nu)
        return self.queues[e].getValueAt(theta)/e.nu + e.tau

    def T(self, e:Edge, theta:ExtendedRational) -> ExtendedRational:
        return theta + self.c(e, theta)

    def pathArrivalTime(self,p:Path,theta:ExtendedRational)->ExtendedRational:
        # Determines the arrival time at the end of path p when starting at time theta
        # TODO: check whether all necessary queues are available
        firstEdge = p.edges[0]
        # TODO: Sobald Path mit leeren Pfaden umgehen kann, hier ebenfalls anpassen
        if len(p) == 1:
            return self.T(firstEdge,theta)
        else:
            return self.pathArrivalTime(Path(p.edges[1:]),self.T(firstEdge,theta))

    def checkFlowConservation(self,v: Node,upTo: ExtendedRational,commodity: int) -> bool:
        theta = ExtendedRational(0)
        while theta < upTo:
            nextTheta = ExtendedRational(1, 0)
            flow = ExtendedRational(0)
            for e in v.incoming_edges:
                flow += self.fMinus[e,commodity].getValueAt(theta)
                nextTheta = min(nextTheta,self.fMinus[e,commodity].getNextStepFrom(theta))
            for e in v.outgoing_edges:
                flow -= self.fPlus[e, commodity].getValueAt(theta)
                nextTheta = min(nextTheta, self.fPlus[e, commodity].getNextStepFrom(theta))
            if flow != 0:
                # TODO: Fehlermeldung
                print("Flow conservation does not hold at node ",v," at time ",theta)
                return False
            theta = nextTheta
        return True

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


    def checkFeasibility(self,upTo: ExtendedRational,sources: List[Node], sinks: List[Node]) -> bool:
        # Check feasibility of the given flow up to the specified time horizon
        # Does not check any conditions at source or sink nodes (TODO)
        # Does not check FIFO (TODO)
        # Does not check non-negativity (TODO?)
        feasible = True
        for i in range(self.noOfCommodities):
            for v in self.network.nodes:
                if not(v == sources[i] or v == sinks[i]):
                    feasible = feasible and self.checkFlowConservation(v,upTo,i)
        for e in self.network.edges:
            feasible = feasible and self.checkQueueAtCap(e,upTo)
            feasible = feasible and self.checkQueue(e,upTo)
        return feasible

    def __str__(self):
        q = ""
        for i in range(self.noOfCommodities):
            q += "Commodity " + str(i) + ":\n"
            for e in self.network.edges:
                q += str(e) + "f+: " + str(self.fPlus[(e,i)]) + "\n"
                q += str(e)+" q: "+str(self.queues[e])+"\n"
                q += str(e) + "f-: " + str(self.fMinus[(e,i)]) + "\n"
            q += "----------------------------------------------------------\n"
        return q

class PartialPathFlow:
    path: Path
    fPlus: List[PWConst]
    fMinus: List[PWConst]

    def __init__(self,path):
        self.path = path
        self.fPlus = []
        self.fMinus = []
        for e in path.edges:
            self.fPlus.append(PWConst([ExtendedRational(0), ExtendedRational(0)], [ExtendedRational(0)]))
            self.fMinus.append(PWConst([ExtendedRational(0), e.tau], [ExtendedRational(0)]))
