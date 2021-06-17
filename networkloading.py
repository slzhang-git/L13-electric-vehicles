from __future__ import annotations

import heapq, math
from typing import List, Dict
from fractions import Fraction


class ExtendedRational(Fraction):
    def __new__(cls,numerator=0,denominator=None,*,_normalize=True):
        if not denominator is None and denominator == 0:
            self = super(Fraction, cls).__new__(cls)
            if numerator > 0:
                self.isInfinite = True
                self._numerator = 1
                return self
            elif numerator < 0:
                self.isInfinite = True
                self._numerator = -1
                return self
            else:
                # TODO
                pass
        else:
            cls.isInfinite = False
            return Fraction.__new__(cls,numerator,denominator)

    def __str__(self):
        if self.isInfinite:
            if self._numerator > 0:
                return "infty"
            else:
                return "-infty"
        else:
            return Fraction.__str__(self)

    def _richcmp(self, other, op):
        if self.isInfinite:
            if self._numerator > 0:
                return op(math.inf,other)
            else:
                return op(-math.inf,other)
        elif isinstance(other,ExtendedRational) and other.isInfinite:
            if other._numerator > 0:
                return op(self,math.inf)
            else:
                return op(self,-math.inf)
        else:
            return Fraction._richcmp(self,other,op)


# A right-constant function
class PWConst:
    noOfSegments: int
    segmentBorders: List[ExtendedRational]
    segmentValues: List[ExtendedRational]
    default: ExtendedRational

    def __init__(self, borders: List[ExtendedRational], values: List[ExtendedRational], default: ExtendedRational=None):
        self.default = default
        assert(len(borders) == len(values)+1)
        self.noOfSegments = len(values)
        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentValues = values

    def addSegment(self,border:ExtendedRational,value:ExtendedRational):
        # Adds a new constant segment at the right side
        assert (self.segmentBorders[-1] <= border)
        self.segmentBorders.append(border)
        self.segmentValues.append(value)
        self.noOfSegments += 1

    def getValueAt(self,x:ExtendedRational) -> ExtendedRational:
        if x < self.segmentBorders[0] or x >= self.segmentBorders[-1]:
            # x is outside the range of the function
            return self.default
        else:
            for i in range(0,self.noOfSegments):
                if x < self.segmentBorders[i+1]:
                    return self.segmentValues[i]

    def getNextStepFrom(self,x:ExtendedRational) -> ExtendedRational:
        if x >= self.segmentBorders[-1]:
            if self.default is None:
                # TODO
                pass
            else:
                return ExtendedRational(1,0)
        else:
            for i in range(0,self.noOfSegments+1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def __str__(self):
        f = "|"+str(self.segmentBorders[0])+"|"
        for i in range(len(self.segmentValues)):
            f += "-"+str(self.segmentValues[i])+"-|"+str(self.segmentBorders[i+1])+"|"
        return f

# A piecewise linear function
class PWLin:
    noOfSegments: int
    segmentBorders: List[ExtendedRational]
    segmentTvalues: List[ExtendedRational]
    segmentMvalues: List[ExtendedRational]

    def __init__(self, borders: List[ExtendedRational], mvalues: List[ExtendedRational], tvalues: List[ExtendedRational]):
        self.noOfSegments = len(mvalues)
        assert (len(tvalues) == len(mvalues))
        assert (len(borders) == self.noOfSegments+1)

        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentMvalues = mvalues
        self.segmentTvalues = tvalues

    def addSegmant(self, border:ExtendedRational, m:ExtendedRational, t:ExtendedRational=None):
        # Adds a new segment on the right side
        # If no t value is provided the function is extended continuously
        if t is None:
            assert (self.noOfSegments > 0)
            t = self.segmentTvalues[-1]+(self.segmentBorders[-1]-self.segmentBorders[-2])*self.segmentMvalues[-1]
        self.segmentBorders.append(border)
        self.segmentMvalues.append(m)
        self.segmentTvalues.append(t)
        self.noOfSegments += 1

    def getValueAt(self,x:ExtendedRational) -> ExtendedRational:
        if x < self.segmentBorders[0] or x > self.segmentBorders[-1]:
            # x is outside the range of the function
            pass
        else:
            for i in range(0,self.noOfSegments):
                if x <= self.segmentBorders[i+1]:
                    return self.segmentTvalues[i] + (x-self.segmentBorders[i])*self.segmentMvalues[i]

    def getNextStepFrom(self,x:ExtendedRational) -> ExtendedRational:
        if x >= self.segmentBorders[-1]:
            if self.default is None:
                # TODO
                pass
            else:
                return ExtendedRational(1,0)
        else:
            for i in range(0,self.noOfSegments+1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def __str__(self):
        f = "|"+str(self.segmentBorders[0])+"|"
        for i in range(len(self.segmentMvalues)):
            f += str(self.segmentTvalues[i])+"-"\
                +str(self.segmentTvalues[i]+(self.segmentBorders[i+1]-self.segmentBorders[i])*self.segmentMvalues[i])\
                +"|"+str(self.segmentBorders[i+1])+"|"

        return f


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



## Our standard example ##
G = Network()
G.addNode("s")
G.addNode("u")
G.addNode("v")
G.addNode("t")
G.addEdge("s","u",ExtendedRational(2),ExtendedRational(1))
G.addEdge("s","u",ExtendedRational(2),ExtendedRational(2))
G.addEdge("u","v",ExtendedRational(1),ExtendedRational(1))
G.addEdge("v","t",ExtendedRational(2),ExtendedRational(1))
G.addEdge("v","t",ExtendedRational(2),ExtendedRational(2))


p1 = Path([G.edges[0],G.edges[2],G.edges[4]])
p2 = Path([G.edges[1],G.edges[2],G.edges[3]])
p3 = Path([G.edges[1],G.edges[2],G.edges[4]])

times = [ExtendedRational(x) for x in [Fraction(0, 1), Fraction(1,1), Fraction(7, 5), Fraction(79,45), Fraction(533, 255), Fraction(60959, 25245), Fraction(87973, 32175), Fraction(645883231, 211679325), Fraction(20352428839, 6044620725), Fraction(102763256971871, 27908013887325), Fraction(38113179216969589, 9535238078169375), Fraction(252728456720421615199, 58613108466507148125), Fraction(370328783636493893629813, 80045968462426595289375), Fraction(9721281192644073548976735839, 1967449858837983285617548125), Fraction(369116333645531547520604449597, 70232387662440863038874784375), Fraction(5884175068934993254068485193929524831, 1056363306091530810182687504415628125), Fraction(10446391101941864475731853564876729086729, 1775150820290273197613917717356077446875), Fraction(56254848400700133856793020388953403224573871711, 9074299395248372174403112441713512428575628125), Fraction(5165072024104986294669764321404308931456585020561269, 792927404989128174552967970344329571862986011609375), Fraction(8516290197799605026078085037864710327099721726814143645279, 1247169348703035064524603072611574620707419219122002828125), Fraction(54627848924376713157711242516891239689502508044128592204519709, 7647678913187031572245700093975596965248675570346738944484375), Fraction(20452983984005701382459891002427076573600020927659196348595271268304479, 2742558324666403802703893523849986072521028627842873059616823122828125), Fraction(1753030505445120871449902478675615618204865152897781826806430171710845158693, 225551492038037662778430473069639526155006136841264738986590966827325859375), Fraction(780334062678853004098073836362886922942396285794809824351187500112712575343076312671, 96495227079759265096427450485578023548840613602655113003123961045966764644178828125), Fraction(4533697632046113227474128485594996397643242695063341986510264003139032886700158307262355317, 539640454727132499427763087184432623836669690941632202337644124122577128407771995419609375), Fraction(473464856597617158264243384159222732196706270921580240486913004779499163850976283431468041414903391, 54321988446693302202514244546409609358625208263495113931942769422144094127472946298750768978828125), Fraction(98860175530477131986318780907914295226180771995537220742860581153709386163760630411528933392439109603419, 10947408375978079621359552846372338465918065125981315402457557059690078148191816048994339411550644140625), Fraction(4572420573369255577988074626279621330446636410240954350771793596412002405052274292127605347180716741285646166132319, 489288984789565548186499672140028525385850578227342579566941251037360071049982725158430398901283433738934098828125), Fraction(32531174468153043288756991271987757733709388502080452242842442729927686867630149169505117162527732444552247356348485464841, 3367756724027001968176886172339967312948382084619768083118224020881166019124476478718170808506623498006153679790023046875), Fraction(703317969722471456069003366506823523580946937592568289609004914444010067693950629891697551608344207732512119349472490344012058986079, 70513974460420357749008951168808110310017429854063667796645691864131103942485392763281527834463065832950878821799993537206098828125), Fraction(15274489640839601790969789082765778387145816080506027841592261649330027815626978704928696269838505869709445905351265386035302899291893335333, 1484584384806571474442622807241802028217243743398270712277726942364309482087626679637013738936562160834663171793595585208389842291005859375), Fraction(1724034282525046518085671722332060292694199854363468971559341772486668424046303335977435213661231896970852137031677171026013801196102824676322523219551, 162594155288574649724883047228538680889469945498562240524092109936320120711900670666569061237285716321891915046310450629833775640818812921321298828125),Fraction(100)]]
#times = [ExtendedRational(0),ExtendedRational(1),ExtendedRational(7,5),ExtendedRational(8,5),ExtendedRational(100),ExtendedRational(200)]
lambdas = [ExtendedRational(x) for x in [2+1.0/2**i for i in range(len(times)-1)]]
#lambdas = [ExtendedRational(3),ExtendedRational(5,2),ExtendedRational(9,4),ExtendedRational(0),ExtendedRational(0)]

f1 = PWConst(times,[l for l in lambdas],ExtendedRational(0))
f2 = PWConst(times,[ExtendedRational(3-l) for l in lambdas],ExtendedRational(0))

flow = networkLoading(G,[(p1,f1),(p2,f2)],ExtendedRational(50))


print(flow)

for theta in times[:6]:
    print("Starting at ", theta, " along path P1: ", flow.pathArrivalTime(p1,theta))
    print("Starting at ", theta, " along path P2: ", flow.pathArrivalTime(p2, theta))
    print("Starting at ", theta, " along path P3: ", flow.pathArrivalTime(p3, theta))

# Check feasibility
print(flow.checkFeasibility(10,[G.nodes[0],G.nodes[0]],[G.nodes[-1],G.nodes[-1]]))


## The adjusted example ##
'''
G = Network()
G.addNode("s")
G.addNode("u")
G.addNode("v")
G.addNode("t")
G.addEdge("s","u",2,1)
G.addEdge("s","u",2,2)
G.addEdge("u","v",1,1)
G.addEdge("v","t",2,2)
G.addEdge("v","t",2,2)


p1 = Path([G.edges[0],G.edges[2],G.edges[4]])
p2 = Path([G.edges[1],G.edges[2],G.edges[3]])
p3 = Path([G.edges[1],G.edges[2],G.edges[4]])

times = [0,2,4,5,10,100,200]
lambdas = [3,2,2,2,2,2]

f1 = PWConst(times,[l for l in lambdas],0)
f2 = PWConst(times,[3-l for l in lambdas],0)

flow = networkLoading(G,[(p1,f1),(p2,f2)],50)

print(flow)

for theta in times[:-2]:
    print("Starting at ", theta, " along path P1: ", flow.pathArrivalTime(p1,theta))
    print("Starting at ", theta, " along path P2: ", flow.pathArrivalTime(p2, theta))
    print("Starting at ", theta, " along path P3: ", flow.pathArrivalTime(p3, theta))
'''