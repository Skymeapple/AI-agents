"""Probability distributions that remember where their parameters came from.

Standard Monte Carlo libraries take numbers. This one takes numbers *and* their
provenance, and carries it through sampling, so a finished run can answer "which
of these inputs was a guess?" -- the question that separates a model you can act
on from one that merely looks quantitative.

Every distribution exposes ``sample(n, rng) -> np.ndarray``. Sampling is
vectorised: a run draws whole columns at once, which is both fast and what the
sensitivity analysis needs, since it correlates each input column against the
output.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from ..provenance import Estimate, Grade, Provenance


class Distribution(abc.ABC):
    """A sampleable quantity with provenance."""

    provenance: Provenance
    label: str

    @abc.abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` values."""

    @abc.abstractmethod
    def mean(self) -> float:
        """Analytic or near-analytic mean, for quick sanity checks."""

    @property
    def grade(self) -> Grade:
        return self.provenance.grade

    def as_estimate(self) -> Estimate[float]:
        """The distribution's central value, tagged with its provenance."""
        return Estimate(self.mean(), self.provenance, label=self.label)

    def describe(self) -> str:
        return f"{self.label}: {self!s}  <- {self.provenance.describe()}"


@dataclass
class Point(Distribution):
    """A known constant. Still carries provenance -- constants can be wrong too."""

    value: float
    provenance: Provenance
    label: str = "point"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, float(self.value))

    def mean(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return f"{self.value:.4g}"


@dataclass
class PERT(Distribution):
    """Three-point (low / mode / high) distribution, the workhorse here.

    A scaled Beta whose shape follows from the three points. PERT is preferred
    over a triangular distribution for expert ranges because it puts less mass in
    the extreme tails -- an expert's "high" is a plausible bad case, not a hard
    bound, and triangular treats it as far more likely than it is.

    ``lam`` controls confidence in the mode: 4 is the classic value, higher
    concentrates mass around the mode.
    """

    low: float
    mode: float
    high: float
    provenance: Provenance
    label: str = "pert"
    lam: float = 4.0

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(
                f"{self.label}: PERT requires low <= mode <= high, got "
                f"{self.low}, {self.mode}, {self.high}"
            )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        span = self.high - self.low
        if span == 0:
            return np.full(n, float(self.low))
        alpha = 1.0 + self.lam * (self.mode - self.low) / span
        beta = 1.0 + self.lam * (self.high - self.mode) / span
        return self.low + span * rng.beta(alpha, beta, size=n)

    def mean(self) -> float:
        return (self.low + self.lam * self.mode + self.high) / (self.lam + 2.0)

    def __str__(self) -> str:
        return f"PERT({self.low:.4g}, {self.mode:.4g}, {self.high:.4g})"


@dataclass
class LogNormal(Distribution):
    """Right-skewed, strictly positive. The right shape for market sizes.

    Parameterised by the median and a multiplicative spread rather than by mu and
    sigma, because "the median is $2B and it could plausibly be 3x either way" is
    a statement an analyst can actually make, while "sigma = 1.1" is not.

    ``spread`` is the ratio between the 95th percentile and the median.
    """

    median: float
    spread: float
    provenance: Provenance
    label: str = "lognormal"

    def __post_init__(self) -> None:
        if self.median <= 0:
            raise ValueError(f"{self.label}: lognormal median must be positive")
        if self.spread <= 1.0:
            raise ValueError(
                f"{self.label}: spread is a p95/median ratio and must exceed 1.0"
            )

    @property
    def sigma(self) -> float:
        # p95 = median * exp(1.645 * sigma)
        return float(np.log(self.spread) / 1.6448536269514722)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.median * np.exp(rng.normal(0.0, self.sigma, size=n))

    def mean(self) -> float:
        return float(self.median * np.exp(self.sigma**2 / 2.0))

    def __str__(self) -> str:
        return f"LogNormal(median={self.median:.4g}, p95/median={self.spread:.2g})"


@dataclass
class BetaProb(Distribution):
    """A probability in ``[0, 1]``, from a three-point range.

    Used for stage success rates. Converts low/mode/high into Beta shape
    parameters via the PERT construction, so a success rate stated as "24% to 35%,
    most likely 29%" becomes a proper distribution rather than a point estimate
    that hides the real uncertainty in the attrition figure.
    """

    low: float
    mode: float
    high: float
    provenance: Provenance
    label: str = "probability"
    lam: float = 4.0

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("mode", self.mode), ("high", self.high)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.label}: {name}={value} is not a probability")
        if not (self.low <= self.mode <= self.high):
            raise ValueError(f"{self.label}: requires low <= mode <= high")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        span = self.high - self.low
        if span == 0:
            return np.full(n, float(self.low))
        alpha = 1.0 + self.lam * (self.mode - self.low) / span
        beta = 1.0 + self.lam * (self.high - self.mode) / span
        return np.clip(self.low + span * rng.beta(alpha, beta, size=n), 0.0, 1.0)

    def mean(self) -> float:
        return (self.low + self.lam * self.mode + self.high) / (self.lam + 2.0)

    def __str__(self) -> str:
        return f"Beta~({self.low:.2f}, {self.mode:.2f}, {self.high:.2f})"


@dataclass
class Triangular(Distribution):
    """Hard-bounded three-point distribution.

    Use where the bounds are genuinely hard (a physical limit, a contractual cap)
    rather than an expert's plausible range -- that is the case PERT handles better.
    """

    low: float
    mode: float
    high: float
    provenance: Provenance
    label: str = "triangular"

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(f"{self.label}: requires low <= mode <= high")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.high == self.low:
            return np.full(n, float(self.low))
        return rng.triangular(self.low, self.mode, self.high, size=n)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    def __str__(self) -> str:
        return f"Triangular({self.low:.4g}, {self.mode:.4g}, {self.high:.4g})"


def from_three_point(
    spec: dict, provenance: Provenance, label: str, kind: str = "pert"
) -> Distribution:
    """Build a distribution from a ``{low, mode, high}`` mapping.

    Accepts a bare number too, which becomes a :class:`Point` -- convenient when
    a prior is genuinely known rather than ranged.
    """
    if isinstance(spec, (int, float)):
        return Point(float(spec), provenance, label)
    missing = {"low", "mode", "high"} - set(spec)
    if missing:
        raise ValueError(f"{label}: three-point spec missing {sorted(missing)}")
    low, mode, high = float(spec["low"]), float(spec["mode"]), float(spec["high"])
    if kind == "probability":
        return BetaProb(low, mode, high, provenance, label)
    if kind == "triangular":
        return Triangular(low, mode, high, provenance, label)
    return PERT(low, mode, high, provenance, label)
