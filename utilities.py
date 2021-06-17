from __future__ import annotations

from fractions import Fraction
import heapq, math


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