"""Outward-rounded bivariate first/mixed jets for the I2 robust-root gate.

Each real or complex jet carries a value, derivatives with respect to the
selected control coordinates, derivatives with respect to the two calibrated
error variables, and their mixed derivatives.  All coefficients are intervals
from the qualified exact fixed-scale dyadic backend; no binary floating-point
arithmetic occurs in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
TC2_RUNTIME = (EXT / "sq3b_r3_tc1_b3v_r2_cont_r6_r1a" / "repair_a1r" /
               "semantic_r1" / "tc2_int_r1" / "runtime")
sys.path.insert(0, str(TC2_RUNTIME))
from dyadic_interval import CI, DIContext, RI


@dataclass(frozen=True)
class RJet:
    v: RI
    dp: tuple[RI, ...]
    de: tuple[RI, RI]
    cross: tuple[tuple[RI, RI], ...]

    @property
    def ctx(self) -> DIContext:
        return self.v.ctx

    @property
    def n(self) -> int:
        return len(self.dp)

    def _coerce(self, other: object) -> "RJet":
        if isinstance(other, RJet):
            if other.ctx is not self.ctx or other.n != self.n:
                raise ValueError("mixed jet contexts or dimensions")
            return other
        if isinstance(other, (int, Fraction, RI)):
            value = other if isinstance(other, RI) else self.ctx.from_fraction(Fraction(other))
            return real_constant(value, self.n)
        raise TypeError(type(other))

    def __add__(self, other: object) -> "RJet":
        rhs = self._coerce(other)
        return RJet(self.v + rhs.v,
                    tuple(a + b for a, b in zip(self.dp, rhs.dp)),
                    (self.de[0] + rhs.de[0], self.de[1] + rhs.de[1]),
                    tuple((a[0] + b[0], a[1] + b[1])
                          for a, b in zip(self.cross, rhs.cross)))

    __radd__ = __add__

    def __neg__(self) -> "RJet":
        return RJet(-self.v, tuple(-x for x in self.dp),
                    (-self.de[0], -self.de[1]),
                    tuple((-x[0], -x[1]) for x in self.cross))

    def __sub__(self, other: object) -> "RJet":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "RJet":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "RJet":
        rhs = self._coerce(other)
        dp = tuple(self.dp[j] * rhs.v + self.v * rhs.dp[j]
                   for j in range(self.n))
        de = tuple(self.de[e] * rhs.v + self.v * rhs.de[e]
                   for e in range(2))
        cross = tuple(tuple(self.cross[j][e] * rhs.v
                                  + self.dp[j] * rhs.de[e]
                                  + self.de[e] * rhs.dp[j]
                                  + self.v * rhs.cross[j][e]
                                  for e in range(2))
                      for j in range(self.n))
        return RJet(self.v * rhs.v, dp, (de[0], de[1]), cross)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "RJet":
        rhs = self._coerce(other)
        return self * reciprocal_real(rhs)

    def __rtruediv__(self, other: object) -> "RJet":
        return self._coerce(other) / self


@dataclass(frozen=True)
class CJet:
    v: CI
    dp: tuple[CI, ...]
    de: tuple[CI, CI]
    cross: tuple[tuple[CI, CI], ...]

    @property
    def ctx(self) -> DIContext:
        return self.v.ctx

    @property
    def n(self) -> int:
        return len(self.dp)

    def _coerce(self, other: object) -> "CJet":
        if isinstance(other, CJet):
            if other.ctx is not self.ctx or other.n != self.n:
                raise ValueError("mixed jet contexts or dimensions")
            return other
        if isinstance(other, RJet):
            return complex_from_real(other)
        if isinstance(other, (int, Fraction, RI, CI)):
            value = other if isinstance(other, CI) else self.ctx.complex(other)
            return complex_constant(value, self.n)
        raise TypeError(type(other))

    def __add__(self, other: object) -> "CJet":
        rhs = self._coerce(other)
        return CJet(self.v + rhs.v,
                    tuple(a + b for a, b in zip(self.dp, rhs.dp)),
                    (self.de[0] + rhs.de[0], self.de[1] + rhs.de[1]),
                    tuple((a[0] + b[0], a[1] + b[1])
                          for a, b in zip(self.cross, rhs.cross)))

    __radd__ = __add__

    def __neg__(self) -> "CJet":
        return CJet(-self.v, tuple(-x for x in self.dp),
                    (-self.de[0], -self.de[1]),
                    tuple((-x[0], -x[1]) for x in self.cross))

    def __sub__(self, other: object) -> "CJet":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "CJet":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "CJet":
        rhs = self._coerce(other)
        dp = tuple(self.dp[j] * rhs.v + self.v * rhs.dp[j]
                   for j in range(self.n))
        de = tuple(self.de[e] * rhs.v + self.v * rhs.de[e]
                   for e in range(2))
        cross = tuple(tuple(self.cross[j][e] * rhs.v
                                  + self.dp[j] * rhs.de[e]
                                  + self.de[e] * rhs.dp[j]
                                  + self.v * rhs.cross[j][e]
                                  for e in range(2))
                      for j in range(self.n))
        return CJet(self.v * rhs.v, dp, (de[0], de[1]), cross)

    __rmul__ = __mul__

    def conjugate(self) -> "CJet":
        return CJet(self.v.conjugate(), tuple(x.conjugate() for x in self.dp),
                    (self.de[0].conjugate(), self.de[1].conjugate()),
                    tuple((x[0].conjugate(), x[1].conjugate()) for x in self.cross))


def real_constant(value: RI, n: int) -> RJet:
    z = value.ctx.zero
    return RJet(value, (z,) * n, (z, z), ((z, z),) * n)


def real_control(value: RI, n: int, coordinate: int | None,
                 error_channel: int | None) -> RJet:
    """Jet for p*(1+epsilon_e), or p when error_channel is None."""
    ctx = value.ctx
    z, o = ctx.zero, ctx.one
    dp = tuple(o if coordinate == j else z for j in range(n))
    de = tuple(value if error_channel == e else z for e in range(2))
    cross = tuple(tuple(o if coordinate == j and error_channel == e else z
                              for e in range(2)) for j in range(n))
    return RJet(value, dp, (de[0], de[1]), cross)


def complex_constant(value: CI, n: int) -> CJet:
    z = value.ctx.complex()
    return CJet(value, (z,) * n, (z, z), ((z, z),) * n)


def complex_from_real(real: RJet, imag: RJet | None = None) -> CJet:
    if imag is None:
        imag = real_constant(real.ctx.zero, real.n)
    if real.ctx is not imag.ctx or real.n != imag.n:
        raise ValueError("mixed real jets")
    return CJet(CI(real.v, imag.v),
                tuple(CI(real.dp[j], imag.dp[j]) for j in range(real.n)),
                (CI(real.de[0], imag.de[0]), CI(real.de[1], imag.de[1])),
                tuple((CI(real.cross[j][0], imag.cross[j][0]),
                       CI(real.cross[j][1], imag.cross[j][1]))
                      for j in range(real.n)))


def reciprocal_real(x: RJet) -> RJet:
    inv = x.v.reciprocal()
    first = -(inv * inv)
    second = 2 * inv * inv * inv
    dp = tuple(first * value for value in x.dp)
    de = tuple(first * value for value in x.de)
    cross = tuple(tuple(second * x.dp[j] * x.de[e] + first * x.cross[j][e]
                              for e in range(2)) for j in range(x.n))
    return RJet(inv, dp, (de[0], de[1]), cross)


def sin_cos_jet(x: RJet) -> tuple[RJet, RJet]:
    sine, cosine = x.ctx.sin_cos(x.v)
    sin_dp = tuple(cosine * value for value in x.dp)
    cos_dp = tuple(-sine * value for value in x.dp)
    sin_de = tuple(cosine * value for value in x.de)
    cos_de = tuple(-sine * value for value in x.de)
    sin_cross = tuple(tuple(-sine * x.dp[j] * x.de[e] + cosine * x.cross[j][e]
                                  for e in range(2)) for j in range(x.n))
    cos_cross = tuple(tuple(-cosine * x.dp[j] * x.de[e] - sine * x.cross[j][e]
                                  for e in range(2)) for j in range(x.n))
    return (RJet(sine, sin_dp, (sin_de[0], sin_de[1]), sin_cross),
            RJet(cosine, cos_dp, (cos_de[0], cos_de[1]), cos_cross))


def exp_minus_i_jet(x: RJet) -> CJet:
    sine, cosine = sin_cos_jet(x)
    return complex_from_real(cosine, -sine)
