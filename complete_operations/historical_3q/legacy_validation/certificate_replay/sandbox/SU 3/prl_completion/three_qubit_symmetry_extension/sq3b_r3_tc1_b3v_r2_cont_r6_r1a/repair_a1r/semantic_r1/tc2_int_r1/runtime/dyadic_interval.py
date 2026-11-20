"""Small exact dyadic interval backend for TC2-INT.

Every endpoint is an integer multiple of 2**(-bits).  Algebraic operations use
integer floor/ceiling rounding.  sin/cos are reduced with an exact rational
Machin enclosure of pi and enclosed by rational Taylor remainders before final
dyadic outward rounding.  No binary float or machine complex is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _nearest_fraction(value: Fraction) -> int:
    if value >= 0:
        return _floor_fraction(value + Fraction(1, 2))
    return _ceil_fraction(value - Fraction(1, 2))


@dataclass(frozen=True)
class RI:
    ctx: "DIContext"
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    def _coerce(self, other: object) -> "RI":
        if isinstance(other, RI):
            if other.ctx is not self.ctx:
                raise ValueError("mixed interval contexts")
            return other
        if isinstance(other, int):
            return self.ctx.point_scaled(other * self.ctx.scale)
        if isinstance(other, Fraction):
            return self.ctx.from_fraction(other)
        raise TypeError(type(other))

    def __add__(self, other: object) -> "RI":
        rhs = self._coerce(other)
        return RI(self.ctx, self.lo + rhs.lo, self.hi + rhs.hi)

    __radd__ = __add__

    def __neg__(self) -> "RI":
        return RI(self.ctx, -self.hi, -self.lo)

    def __sub__(self, other: object) -> "RI":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "RI":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "RI":
        rhs = self._coerce(other)
        products = (self.lo * rhs.lo, self.lo * rhs.hi,
                    self.hi * rhs.lo, self.hi * rhs.hi)
        scale = self.ctx.scale
        return RI(self.ctx, min(products) // scale,
                  -((-max(products)) // scale))

    __rmul__ = __mul__

    def reciprocal(self) -> "RI":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("interval contains zero")
        ss = self.ctx.scale * self.ctx.scale
        return RI(self.ctx, ss // self.hi, -((-ss) // self.lo))

    def __truediv__(self, other: object) -> "RI":
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: object) -> "RI":
        return self._coerce(other) / self

    def square(self) -> "RI":
        if self.lo <= 0 <= self.hi:
            upper = max(self.lo * self.lo, self.hi * self.hi)
            return RI(self.ctx, 0, -((-upper) // self.ctx.scale))
        return self * self

    def abs(self) -> "RI":
        if self.lo <= 0 <= self.hi:
            return RI(self.ctx, 0, max(-self.lo, self.hi))
        return RI(self.ctx, min(abs(self.lo), abs(self.hi)),
                  max(abs(self.lo), abs(self.hi)))

    def lower_fraction(self) -> Fraction:
        return Fraction(self.lo, self.ctx.scale)

    def upper_fraction(self) -> Fraction:
        return Fraction(self.hi, self.ctx.scale)

    def width_fraction(self) -> Fraction:
        return Fraction(self.hi - self.lo, self.ctx.scale)

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi


@dataclass(frozen=True)
class CI:
    re: RI
    im: RI

    def __post_init__(self) -> None:
        if self.re.ctx is not self.im.ctx:
            raise ValueError("mixed complex interval contexts")

    @property
    def ctx(self) -> "DIContext":
        return self.re.ctx

    def _coerce(self, other: object) -> "CI":
        if isinstance(other, CI):
            if other.ctx is not self.ctx:
                raise ValueError("mixed interval contexts")
            return other
        if isinstance(other, (int, Fraction, RI)):
            real = other if isinstance(other, RI) else self.ctx.from_fraction(Fraction(other))
            return CI(real, self.ctx.zero)
        raise TypeError(type(other))

    def __add__(self, other: object) -> "CI":
        rhs = self._coerce(other)
        return CI(self.re + rhs.re, self.im + rhs.im)

    __radd__ = __add__

    def __neg__(self) -> "CI":
        return CI(-self.re, -self.im)

    def __sub__(self, other: object) -> "CI":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "CI":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "CI":
        rhs = self._coerce(other)
        return CI(self.re * rhs.re - self.im * rhs.im,
                  self.re * rhs.im + self.im * rhs.re)

    __rmul__ = __mul__

    def conjugate(self) -> "CI":
        return CI(self.re, -self.im)

    def __truediv__(self, other: object) -> "CI":
        rhs = self._coerce(other)
        denominator = rhs.re.square() + rhs.im.square()
        if denominator.contains_zero():
            raise ZeroDivisionError("complex interval denominator may vanish")
        numerator = self * rhs.conjugate()
        return CI(numerator.re / denominator, numerator.im / denominator)

    def abs(self) -> RI:
        return self.ctx.sqrt(self.re.square() + self.im.square())


class DIContext:
    def __init__(self, bits: int):
        if bits < 128:
            raise ValueError("TC2 requires at least 128 dyadic bits")
        self.bits = int(bits)
        self.scale = 1 << self.bits
        self.zero = RI(self, 0, 0)
        self.one = RI(self, self.scale, self.scale)
        self.pi = self._build_pi()

    def point_scaled(self, value: int) -> RI:
        return RI(self, int(value), int(value))

    def point_nearest(self, value: Fraction) -> RI:
        scaled = _nearest_fraction(value * self.scale)
        return RI(self, scaled, scaled)

    def from_fraction(self, value: Fraction) -> RI:
        scaled = value * self.scale
        return RI(self, _floor_fraction(scaled), _ceil_fraction(scaled))

    def from_bounds(self, lower: Fraction, upper: Fraction) -> RI:
        if lower > upper:
            raise ValueError("reversed rational bounds")
        return RI(self, _floor_fraction(lower * self.scale),
                  _ceil_fraction(upper * self.scale))

    def decimal(self, text: str) -> RI:
        return self.from_fraction(Fraction(text))

    def complex(self, real: RI | int | Fraction = 0,
                imag: RI | int | Fraction = 0) -> CI:
        re = real if isinstance(real, RI) else self.from_fraction(Fraction(real))
        im = imag if isinstance(imag, RI) else self.from_fraction(Fraction(imag))
        return CI(re, im)

    def sqrt(self, value: RI) -> RI:
        if value.lo < 0:
            raise ValueError("sqrt of negative interval")
        lo_arg = value.lo * self.scale
        hi_arg = value.hi * self.scale
        lo = isqrt(lo_arg)
        hi = isqrt(hi_arg)
        if hi * hi != hi_arg:
            hi += 1
        return RI(self, lo, hi)

    def _atan_bounds(self, reciprocal: int) -> tuple[Fraction, Fraction]:
        x = Fraction(1, reciprocal)
        total = Fraction(0)
        n = 0
        target = Fraction(1, 1 << (self.bits + 40))
        while True:
            term = x ** (2 * n + 1) / (2 * n + 1)
            total = total + term if n % 2 == 0 else total - term
            next_term = x ** (2 * n + 3) / (2 * n + 3)
            if next_term < target:
                if (n + 1) % 2 == 0:
                    lower, upper = total, total + next_term
                else:
                    lower, upper = total - next_term, total
                return lower, upper
            n += 1

    def _build_pi(self) -> RI:
        a_lo, a_hi = self._atan_bounds(5)
        b_lo, b_hi = self._atan_bounds(239)
        return self.from_bounds(16 * a_lo - 4 * b_hi,
                                16 * a_hi - 4 * b_lo)

    def _point_sin_bounds(self, x: Fraction) -> tuple[Fraction, Fraction]:
        term = x
        total = term
        n = 0
        target = Fraction(1, 1 << (self.bits + 40))
        while True:
            next_term = -term * x * x / ((2 * n + 2) * (2 * n + 3))
            if abs(next_term) < target:
                return total - abs(next_term), total + abs(next_term)
            total += next_term
            term = next_term
            n += 1

    def _point_cos_bounds(self, x: Fraction) -> tuple[Fraction, Fraction]:
        term = Fraction(1)
        total = term
        n = 0
        target = Fraction(1, 1 << (self.bits + 40))
        while True:
            next_term = -term * x * x / ((2 * n + 1) * (2 * n + 2))
            if abs(next_term) < target:
                return total - abs(next_term), total + abs(next_term)
            total += next_term
            term = next_term
            n += 1

    def _reduced_sin_cos(self, y: RI) -> tuple[RI, RI]:
        lo = y.lower_fraction()
        hi = y.upper_fraction()
        sin_lo = self._point_sin_bounds(lo)[0]
        sin_hi = self._point_sin_bounds(hi)[1]
        sin_y = self.from_bounds(sin_lo, sin_hi)

        if lo >= 0:
            cos_lower = self._point_cos_bounds(hi)[0]
            cos_upper = self._point_cos_bounds(lo)[1]
        elif hi <= 0:
            cos_lower = self._point_cos_bounds(lo)[0]
            cos_upper = self._point_cos_bounds(hi)[1]
        else:
            left_lower = self._point_cos_bounds(lo)[0]
            right_lower = self._point_cos_bounds(hi)[0]
            cos_lower = min(left_lower, right_lower)
            cos_upper = Fraction(1)
        return sin_y, self.from_bounds(cos_lower, cos_upper)

    def sin_cos(self, value: RI) -> tuple[RI, RI]:
        midpoint = Fraction(value.lo + value.hi, 2 * self.scale)
        pi_mid = Fraction(self.pi.lo + self.pi.hi, 2 * self.scale)
        k = _nearest_fraction(midpoint / (pi_mid / 2))
        y = value - self.pi * Fraction(k, 2)
        # The monotonic endpoint rules below only need the reduced interval to
        # stay inside (-pi/2, pi/2).  Use the *inner* half-pi bound so this
        # domain check is itself rigorous rather than relying on an outer
        # approximation to pi/4.
        half_pi = self.pi / 2
        if y.lo < -half_pi.lo or y.hi > half_pi.lo:
            raise ValueError("range reduction did not reach (-pi/2,pi/2)")
        sin_y, cos_y = self._reduced_sin_cos(y)
        quadrant = k % 4
        if quadrant == 0:
            return sin_y, cos_y
        if quadrant == 1:
            return cos_y, -sin_y
        if quadrant == 2:
            return -sin_y, -cos_y
        return -cos_y, sin_y

    def sin(self, value: RI) -> RI:
        return self.sin_cos(value)[0]

    def cos(self, value: RI) -> RI:
        return self.sin_cos(value)[1]

    def exp_minus_i(self, value: RI) -> CI:
        sine, cosine = self.sin_cos(value)
        return CI(cosine, -sine)
