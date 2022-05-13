#!/bin/bash
time python3 experiments.py evExample6Edges.txt evExample6Comm.txt example6 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6_0.25
time python3 experiments.py evExample6EdgesWR.txt evExample6Comm.txt example6WR 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6WR_0.25

time python3 experiments.py evExample6Edges.txt evExample6Comm.txt example6 50 20000 900 0.001 0.75 0.125 1 > exampleOutputs/example6_0.125
time python3 experiments.py evExample6EdgesWR.txt evExample6Comm.txt example6WR 50 20000 900 0.001 0.75 0.125 1 > exampleOutputs/example6WR_0.125

#time python3 experiments.py evExample6Edges.txt evExample6Comm.txt example6 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6_0.25
#time python3 experiments.py evExample6EdgesWR.txt evExample6Comm.txt example6WR 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6WR_0.25

#time python3 experiments.py evExample6Edges.txt evExample6Comm.txt example6 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6_0.25
#time python3 experiments.py evExample6EdgesWR.txt evExample6Comm.txt example6WR 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6WR_0.25

#time python3 experiments.py evExample6Edges.txt evExample6Comm.txt example6 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6_0.25
#time python3 experiments.py evExample6EdgesWR.txt evExample6Comm.txt example6WR 50 20000 900 0.001 0.75 0.25 1 > exampleOutputs/example6WR_0.25
