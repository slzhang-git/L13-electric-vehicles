from __future__ import annotations
from typing import List

from fractions import Fraction
import heapq, math

import matplotlib.pyplot as plt


# A class for rational numbers including +/- infinity
# (basically just the class Fraction with +/- infinity)
class ExtendedRational(Fraction):
    def __new__(cls, numerator=0, denominator=None, *, _normalize=True):
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
            return Fraction.__new__(cls, numerator, denominator)

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
                return op(math.inf, other)
            else:
                return op(-math.inf, other)
        elif isinstance(other, ExtendedRational) and other.isInfinite:
            if other._numerator > 0:
                return op(self, math.inf)
            else:
                return op(self, -math.inf)
        else:
            return Fraction._richcmp(self, other, op)


# A right-constant function
class PWConst:
    noOfSegments: int
    segmentBorders: List[ExtendedRational]
    segmentValues: List[ExtendedRational]
    default: ExtendedRational
    autoSimplify: bool

    def __init__(self, borders: List[ExtendedRational], values: List[ExtendedRational],
                 default: ExtendedRational = None, autoSimplyfy: bool = True):
        # autoSimplify=True means that adjacent segments with the same value are automatically unified to a single segment
        self.default = default
        assert (len(borders) == len(values) + 1)
        self.noOfSegments = len(values)
        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentValues = values

        self.autoSimplify = autoSimplyfy

    def addSegment(self, border: ExtendedRational, value: ExtendedRational):
        # Adds a new constant segment at the right side
        assert (self.segmentBorders[-1] <= border)

        if self.autoSimplify and len(self.segmentValues) > 0 and self.segmentValues[-1] == value:
            self.segmentBorders[-1] = border
        else:
            self.segmentBorders.append(border)
            self.segmentValues.append(value)
            self.noOfSegments += 1

    def getValueAt(self, x: ExtendedRational) -> ExtendedRational:
        if x < self.segmentBorders[0] or x >= self.segmentBorders[-1]:
            # x is outside the range of the function
            return self.default
        else:
            for i in range(0, self.noOfSegments):
                if x < self.segmentBorders[i + 1]:
                    return self.segmentValues[i]

    def getNextStepFrom(self, x: ExtendedRational) -> ExtendedRational:
        if x >= self.segmentBorders[-1]:
            if self.default is None:
                # TODO
                pass
            else:
                return ExtendedRational(1, 0)
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def __add__(self, other: PWConst) -> PWConst:
        # Add to piecewise constant functions
        # TODO: Depends on which function does not have default...
        if self.default is None or other.default is None:
            default = None
            leftMost = max(self.segmentBorders[0], other.segmentBorders[0])
            rightMost = min(self.segmentBorders[-1], other.segmentBorders[-1])
        else:
            default = self.default + other.default
            leftMost = min(self.segmentBorders[0], other.segmentBorders[0])
            rightMost = max(self.segmentBorders[-1], other.segmentBorders[-1])

        sum = PWConst([leftMost], [], default, self.autoSimplify and other.autoSimplify)

        x = leftMost
        while x < rightMost:
            val = self.getValueAt(x) + other.getValueAt(x)
            x = min(self.getNextStepFrom(x), other.getNextStepFrom(x))
            sum.addSegment(x, val)

        return sum

    def smul(self, mu: ExtendedRational) -> PWConst:
        # Creates a new piecewise constant function by scaling the current one by mu

        if self.default is None:
            default = None
        else:
            default = mu * self.default
        scaled = PWConst([self.segmentBorders[0]], [], default, self.autoSimplify)
        for i in range(len(self.segmentValues)):
            scaled.addSegment(self.segmentBorders[i + 1], mu * self.segmentValues[i])

        return scaled

    def restrictTo(self, a: ExtendedRational, b: ExtendedRational, default: ExtendedRational = None) -> PWConst:
        # Creates a new piecewise constant function by restricting the current one to the interval [a,b]
        restrictedF = PWConst([a], [], default=default)
        x = a
        while x <= self.segmentBorders[-1] and x < b:
            val = self.getValueAt(x)
            x = min(self.getNextStepFrom(x), b)
            restrictedF.addSegment(x, val)
        if x < b and (not self.getValueAt(x) is None):
            restrictedF.addSegment(b, self.getValueAt(x))
        return restrictedF

    def __abs__(self) -> PWConst:
        # Creates a new piecewise constant functions |f|
        if self.default is None:
            default = None
        else:
            default = self.default.__abs__()
        absf = PWConst([self.segmentBorders[0]], [], default, self.autoSimplify)
        for i in range(len(self.segmentValues)):
            absf.addSegment(self.segmentBorders[i + 1], self.segmentValues[i].__abs__())

        return absf

    def integrate(self, a: ExtendedRational, b: ExtendedRational) -> ExtendedRational:
        # Determines the value of the integral of the given piecewise function from a to b
        assert (self.default is not None or (a >= self.segmentBorders[0] and b <= self.segmentBorders[-1]))

        integral = ExtendedRational(0)
        x = a
        while x < b:
            y = min(self.getNextStepFrom(x), b)
            integral += (y - x) * self.getValueAt(x)
            x = y

        return integral

    def norm(self) -> ExtendedRational:
        # Computes the L1-norm
        assert (self.default is None or self.default == 0)
        return self.__abs__().integrate(self.segmentBorders[0], self.segmentBorders[-1])

    def drawGraph(self, start: ExtendedRational, end: ExtendedRational):
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

    def __str__(self):
        f = "|" + str(self.segmentBorders[0]) + "|"
        for i in range(len(self.segmentValues)):
            f += "-" + str(self.segmentValues[i]) + "-|" + str(self.segmentBorders[i + 1]) + "|"
        return f


# A piecewise linear function
class PWLin:
    noOfSegments: int
    autoSimplify: bool
    segmentBorders: List[ExtendedRational]
    segmentTvalues: List[ExtendedRational]
    segmentMvalues: List[ExtendedRational]

    def __init__(self, borders: List[ExtendedRational], mvalues: List[ExtendedRational],
                 tvalues: List[ExtendedRational], autoSimplify: bool = True):
        # autoSimplify=True means that adjacent segments are automatically unified whenever possible
        self.autoSimplify = autoSimplify

        self.noOfSegments = len(mvalues)
        assert (len(tvalues) == len(mvalues))
        assert (len(borders) == self.noOfSegments + 1)

        # TODO: check that borders are non-decreasing
        self.segmentBorders = borders
        self.segmentMvalues = mvalues
        self.segmentTvalues = tvalues

    def addSegmant(self, border: ExtendedRational, m: ExtendedRational, t: ExtendedRational = None):
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

    def getValueAt(self, x: ExtendedRational) -> ExtendedRational:
        if x < self.segmentBorders[0] or x > self.segmentBorders[-1]:
            # x is outside the range of the function
            pass
        else:
            for i in range(0, self.noOfSegments):
                if x <= self.segmentBorders[i + 1]:
                    return self.segmentTvalues[i] + (x - self.segmentBorders[i]) * self.segmentMvalues[i]

    def getNextStepFrom(self, x: ExtendedRational) -> ExtendedRational:
        if x >= self.segmentBorders[-1]:
            if self.default is None:
                # TODO
                pass
            else:
                return ExtendedRational(1, 0)
        else:
            for i in range(0, self.noOfSegments + 1):
                if x < self.segmentBorders[i]:
                    return self.segmentBorders[i]

    def drawGraph(self, start: ExtendedRational, end: ExtendedRational):
        x = [start]
        y = [self.getValueAt(start)]
        while self.getNextStepFrom(x[-1]) < end:
            x.append(self.getNextStepFrom(x[-1]))
            y.append(self.getValueAt(x[-1]))
        x.append(end)
        y.append(self.getValueAt(end))
        plt.plot(x, y)
        return plt

    def __str__(self):
        f = "|" + str(self.segmentBorders[0]) + "|"
        for i in range(len(self.segmentMvalues)):
            f += str(self.segmentTvalues[i]) + "-" \
                 + str(
                self.segmentTvalues[i] + (self.segmentBorders[i + 1] - self.segmentBorders[i]) * self.segmentMvalues[i]) \
                 + "|" + str(self.segmentBorders[i + 1]) + "|"

        return f
