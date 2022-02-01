from __future__ import annotations

import math
from typing import List

from fractions import Fraction


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

    def __float__(self) -> float:
        if self.isInfinite:
            if self.numerator > 0:
                return float(math.inf)
            elif self.numerator < 0:
                return float(-math.inf)
            else:
                return float(0/0)
        else:
            return Fraction.__float__(self)



# If precision = 0, we use floating point numbers for all calculaion
# Otherwise we use extendedRationals
precision = 1
#number = float
number = ExtendedRational

if precision == 0:
    infinity = math.inf
    zero = 0.0
    def makeNumber(n):
        return float(n)
else:
    infinity = ExtendedRational(1,0)
    zero = ExtendedRational(0)
    def makeNumber(n):
        return ExtendedRational(n)

