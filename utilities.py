from __future__ import annotations
from typing import List, Tuple

import matplotlib.pyplot as plt

from extendedRational import *

# A right-constant function with finitely many steps
class PWConst:
    noOfSegments: int
    segmentBorders: List[number]
    segmentValues: List[number]
    defaultValue: number
    autoSimplify: bool

    def __init__(self, borders: List[number], values: List[number],
                 defaultValue: number = None, autoSimplyfy: bool = True):
        # autoSimplify=True means that adjacent segments with the same value are automatically unified
        # to a single segment
        # If a defaultValue is given this is the value of the function outside of the given borders
        # If none is given the function is undefined there
        self.defaultValue = defaultValue
        assert (len(borders) == len(values) + 1)
        self.noOfSegments = len(values)
        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentValues = values

        self.autoSimplify = autoSimplyfy

    def addSegment(self, border: number, value: number):
        # Adds a new constant segment at the right side
        assert (self.segmentBorders[-1] <= border + numPrecision)

        # Only add new segment if it is larger than the given precision
        if self.segmentBorders[-1] - numPrecision < border < self.segmentBorders[-1]+numPrecision:
            return

        if self.autoSimplify and len(self.segmentValues) > 0 and self.segmentValues[-1] == value:
            self.segmentBorders[-1] = border
        else:
            self.segmentBorders.append(border)
            self.segmentValues.append(value)
            self.noOfSegments += 1

    def getValueAt(self, x: number) -> number:
        # Returns the value of the function at x
        if x < self.segmentBorders[0] or x >= self.segmentBorders[-1]:
            # x is outside the range of the function
            return self.defaultValue
        else:
            for i in range(0, self.noOfSegments):
                if x < self.segmentBorders[i + 1]:
                    return self.segmentValues[i]

    def getNextStepFrom(self, x: number) -> number:
        # get the next step of the function strictly after x
        if x >= self.segmentBorders[-1]:
            if self.defaultValue is None:
                # TODO: Raise an error
                pass
            else:
                return infinity
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def __add__(self, other: PWConst) -> PWConst:
        # Add two piecewise constant functions

        # If at least one of the functions is undefined outside its borders the sum of the two function can only be
        # defined within the boundaries of that function (the intersection of the two boundaries if both functions
        # are undefined outside their boundaries)
        if self.defaultValue is None and other.defaultValue is None:
            default = None
            leftMost = max(self.segmentBorders[0], other.segmentBorders[0])
            rightMost = min(self.segmentBorders[-1], other.segmentBorders[-1])
        elif self.defaultValue is None and not (other.defaultValue is None):
            default = None
            leftMost = self.segmentBorders[0]
            rightMost = self.segmentBorders[-1]
        elif not(self.defaultValue is None) and other.defaultValue is None:
            default = None
            leftMost = other.segmentBorders[0]
            rightMost = other.segmentBorders[-1]
        else:
            default = self.defaultValue + other.defaultValue
            leftMost = min(self.segmentBorders[0], other.segmentBorders[0])
            rightMost = max(self.segmentBorders[-1], other.segmentBorders[-1])

        sum = PWConst([leftMost], [], default, self.autoSimplify and other.autoSimplify)

        x = leftMost
        while x < rightMost:
            val = self.getValueAt(x) + other.getValueAt(x)
            x = min(self.getNextStepFrom(x), other.getNextStepFrom(x))
            sum.addSegment(x, val)

        return sum

    def smul(self, mu: number) -> PWConst:
        # Creates a new piecewise constant function by scaling the current one by mu

        if self.defaultValue is None:
            default = None
        else:
            default = mu * self.defaultValue
        scaled = PWConst([self.segmentBorders[0]], [], default, self.autoSimplify)
        for i in range(len(self.segmentValues)):
            scaled.addSegment(self.segmentBorders[i + 1], mu * self.segmentValues[i])

        return scaled

    def restrictTo(self, a: number, b: number, default: number = None) -> PWConst:
        # Creates a new piecewise constant function by restricting the current one to the interval [a,b)
        # and setting it to default outside [a,b)

        x = max(a,self.segmentBorders[0])
        restrictedF = PWConst([x], [], defaultValue=default)

        while x <= self.segmentBorders[-1] and x < b:
            val = self.getValueAt(x)
            x = min(self.getNextStepFrom(x), b)
            restrictedF.addSegment(x, val)

        if x < b and (not self.getValueAt(x) is None):
            restrictedF.addSegment(b, self.getValueAt(x))

        return restrictedF

    def isZero(self) -> bool:
        # Checks whether the function is zero wherever it is defined
        if self.defaultValue is not None and -infinity < self.segmentBorders[0] \
                and self.segmentBorders[-1] < infinity and self.defaultValue != zero:
            # If the default value is not zero, the function is not zero
            return False
        for y in self.segmentValues:
            if y != zero:
                # If there is one segment where the function is non-zero, the function is not zero
                # (this assume that there are no zero-length intervals!)
                return False
        return True

    def __abs__(self) -> PWConst:
        # Creates a new piecewise constant functions |f|
        if self.defaultValue is None:
            default = None
        else:
            default = self.defaultValue.__abs__()
        absf = PWConst([self.segmentBorders[0]], [], default, self.autoSimplify)
        for i in range(len(self.segmentValues)):
            absf.addSegment(self.segmentBorders[i + 1], self.segmentValues[i].__abs__())

        return absf

    def integrate(self, a: number, b: number) -> number:
        # Determines the value of the integral of the given piecewise function from a to b
        assert (self.defaultValue is not None or (a >= self.segmentBorders[0] and b <= self.segmentBorders[-1]))

        integral = zero
        x = a
        while x < b:
            y = min(self.getNextStepFrom(x), b)
            integral += (y - x) * self.getValueAt(x)
            x = y

        return integral

    def norm(self) -> number:
        # Computes the L1-norm of the function
        # requires the function to either be undefined or zero outside its borders
        # (otherwise the L1-norm would be undefined/+-infty)
        assert (self.defaultValue is None or self.defaultValue == 0)
        return self.__abs__().integrate(self.segmentBorders[0], self.segmentBorders[-1])

    def drawGraph(self, start: number, end: number):
        # Draws a graph of the function between start and end
        current = start
        x = []
        y = []
        while self.getNextStepFrom(current) < end:
            x.append(current)
            x.append(self.getNextStepFrom(current))
            y.append(self.getValueAt(current))
            y.append(self.getValueAt(current))
            current = self.getNextStepFrom(current)
        x.append(current)
        x.append(end)
        y.append(self.getValueAt(current))
        y.append(self.getValueAt(current))
        plt.plot(x, y)
        return plt

    def getXandY(self, start: number, end: number) -> Tuple[List[number],List[number]]:
        # Returns two vectors x and y representing the function between start and end in the following form:
        # x = [a_0,a_1,a_1,a_2,a_2,...,a_n], y = [b_0,b_0,b_1,b_1,...,b_{n-1}]
        # such that [a_i,a_{i+1}) form a partition of [start,nextStep(end))
        # into maximal (if autoSimplify=True) intervals of constant value b_i of the function
        # i.e. for even i x[i] is the left boundary of such an interval and y[i] the value in it
        #      for odd i  x[i] is the right boundary of such an interval and y[i] the value in it
        current = start
        x = []
        y = []
        while self.getNextStepFrom(current) < end:
            x.append(current)
            x.append(self.getNextStepFrom(current))
            y.append(self.getValueAt(current))
            y.append(self.getValueAt(current))
            current = self.getNextStepFrom(current)
        x.append(current)
        x.append(end)
        y.append(self.getValueAt(current))
        y.append(self.getValueAt(current))
        return x,y

    def __str__(self):
        f = "|" + str(round(float(self.segmentBorders[0]),2)) + "|"
        for i in range(len(self.segmentValues)):
            # f += "-" + str(self.segmentValues[i]) + "-|" + str(self.segmentBorders[i + 1]) + "|"
            f += " " + str(round(float(self.segmentValues[i]),2)) + " |"
            if self.segmentBorders[i+1] < infinity:
                f += str(round(float(self.segmentBorders[i + 1]),2)) + "|"
            else:
                f += str(self.segmentBorders[i + 1]) + "|"
        return f


# A piecewise linear function with finitely many break points
# I.e. each piece of the function is of the form f(x) = mx + t for all x in some interval (a,b]
class PWLin:
    noOfSegments: int
    autoSimplify: bool
    segmentBorders: List[number]
    segmentTvalues: List[number]
    segmentMvalues: List[number]

    def __init__(self, borders: List[number], mvalues: List[number],
                 tvalues: List[number], autoSimplify: bool = True):
        # autoSimplify=True means that adjacent segments are automatically unified whenever possible
        self.autoSimplify = autoSimplify

        self.noOfSegments = len(mvalues)
        assert (len(tvalues) == len(mvalues))
        assert (len(borders) == self.noOfSegments + 1)

        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentMvalues = mvalues
        self.segmentTvalues = tvalues

    def addSegmant(self, border: number, m: number, t: number = None):
        # Adds a new segment on the right side
        # If no t value is provided the function is extended continuously
        if t is None:
            assert (self.noOfSegments > 0)
            t = self.segmentTvalues[-1] + (self.segmentBorders[-1] - self.segmentBorders[-2]) * self.segmentMvalues[-1]

        if self.autoSimplify and self.noOfSegments > 0 and self.segmentMvalues[-1] == m and self.getValueAt(
                self.segmentBorders[-1]) == t:
            self.segmentBorders[-1] = border
        else:
            self.segmentBorders.append(border)
            self.segmentMvalues.append(m)
            self.segmentTvalues.append(t)
            self.noOfSegments += 1

    def getValueAt(self, x: number) -> number:
        # Returns the value of the function at x
        if x < self.segmentBorders[0] or x > self.segmentBorders[-1]:
            # x is outside the range of the function
            pass # TODO: Raise error
        else:
            for i in range(0, self.noOfSegments):
                if x <= self.segmentBorders[i + 1]:
                    return self.segmentTvalues[i] + (x - self.segmentBorders[i]) * self.segmentMvalues[i]

    def getNextStepFrom(self, x: number) -> number:
        # Returns the next break point strictly after x
        if x >= self.segmentBorders[-1]:
            # TODO: Implement default value and/or better error handling
            pass
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def drawGraph(self, start: number, end: number):
        # Draws a graph of the function between start and end
        x = [start]
        y = [self.getValueAt(start)]
        while self.getNextStepFrom(x[-1]) < end:
            x.append(self.getNextStepFrom(x[-1]))
            y.append(self.getValueAt(x[-1]))
        x.append(end)
        y.append(self.getValueAt(end))
        plt.plot(x, y)
        return plt

    def segment_as_str(self,i:int,ommitStart:bool=True) -> str:
        # Creates a string of the form |2|3 4|4| for the i-th segment
        # |2| is omitted if ommitStart=True (standard)
        assert(i<self.noOfSegments)
        s = ""
        if not ommitStart:
            s += "|" + str(round(float(self.segmentBorders[i]),2)) + "|"
        s += str(round(float(self.segmentTvalues[i]),2)) + " "
        if self.segmentBorders[i+1] < infinity:
            s += str(round(float(self.segmentTvalues[i] + (self.segmentBorders[i + 1] - self.segmentBorders[i]) * self.segmentMvalues[i]),2))
        else:
            if self.segmentMvalues[i] == 0:
                s += "0"
            elif self.segmentMvalues[i] > 0:
                s += "infty"
            else:
                s += "-infty"
        s += "|" + str(round(float(self.segmentBorders[i+1]),2)) + "|"
        return s

    def __str__(self) -> str:
        f = "|" + str(round(float(self.segmentBorders[0]),2)) + "|"
        for i in range(len(self.segmentMvalues)):
            f += self.segment_as_str(i)

        return f

    # def __str__(self):
        # f = "|" + str(self.segmentBorders[0]) + "|"
        # for i in range(len(self.segmentMvalues)):
            # f += str(self.segmentTvalues[i]) + "-" \
                 # + str(
                # self.segmentTvalues[i] + (self.segmentBorders[i + 1] - self.segmentBorders[i]) * self.segmentMvalues[i]) \
                 # + "|" + str(self.segmentBorders[i + 1]) + "|"

        # return f
