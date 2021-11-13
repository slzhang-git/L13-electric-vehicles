from flows import *
from typing import Callable, Dict, List, Set, Tuple
from priorityqueues import *
from utilities import *

# From Michael Markl: https://github.com/Schedulaar/predicted-dynamic-flows/blob/main/predictor/src/core/dijkstra.py

def dynamic_dijkstra(
        phi: ExtendedRational, source: Node, sink: Node, f : PartialFlow)\
            -> Tuple[Dict[Node, ExtendedRational], Dict[Edge, ExtendedRational]]:
    """
    Assumes costs to follow the FIFO rule and relevant_nodes to contain
    all nodes that lie on a path from source to sink.
    Returns the earliest arrival times when departing from source at
    time phi for nodes that source can reach up to the arrival at sink.
    """
    arrival_times: Dict[Node, ExtendedRational] = {}
    queue: PriorityQueue[Node] = PriorityQueue([(source, phi)])
    realized_cost = {}
    while len(queue) > 0:
        arrival_time, v = queue.min_key(), queue.pop()
        arrival_times[v] = arrival_time
        if v == sink:
            break
        for e in v.outgoing_edges:
            w = e.node_to
            if w in arrival_times.keys():
                continue
            realized_cost[e] = f.c(e,arrival_times[v])
            relaxation = arrival_times[v] + realized_cost[e]
            if not queue.has(w):
                queue.push(w, relaxation)
            elif relaxation < queue.key_of(w):
                queue.decrease_key(w, relaxation)
    return arrival_times, realized_cost