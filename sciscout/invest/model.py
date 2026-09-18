"""Investment case: market assumptions and deal terms.

Split from the Monte Carlo engine deliberately. This module holds *what you
believe*; the engine holds *how it is computed*. You should be able to read an
investment case and argue with it without reading any simulation code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..provenance import Provenance, assumed
from .dist import Distribution, LogNormal, PERT, Point


@dataclass
class MarketModel:
    """What the market looks like if the technology actually works.

    Every field is a distribution, because none of these are known. Market size
    in particular is modelled log-normally: market forecasts are right-skewed and
    the downside is bounded at zero while the upside is not.
    """

    tam_usd: Distribution
    peak_share: Distribution
    gross_margin: Distribution
    ramp_years: Distribution
    exit_revenue_multiple: Distribution
    # Years from market entry to the assumed exit or valuation point.
    years_to_exit: Distribution

    @classmethod
    def placeholder(cls, tam_usd: float = 2.0e9) -> "MarketModel":
        """A clearly-labelled starting point, not a recommendation.

        Everything here is ``ASSUMED`` and flagged for review. It exists so a
        track can be modelled end to end the moment it is ingested, with the
        resulting numbers visibly marked as unfounded until someone supplies a
        real market view.
        """
        note = lambda text: assumed(  # noqa: E731
            f"placeholder market input: {text}. Replace before relying on output.",
            needs_review=True,
        )
        return cls(
            tam_usd=LogNormal(
                tam_usd, 3.0, note("serviceable market size"), "TAM (USD)"
            ),
            peak_share=PERT(0.01, 0.05, 0.20, note("peak market share"), "peak share"),
            gross_margin=PERT(0.25, 0.50, 0.75, note("gross margin"), "gross margin"),
            ramp_years=PERT(2.0, 4.0, 8.0, note("years to reach peak share"), "ramp (y)"),
            exit_revenue_multiple=PERT(
                1.5, 4.0, 10.0, note("EV/revenue at exit"), "exit revenue multiple"
            ),
            years_to_exit=PERT(
                3.0, 5.0, 9.0, note("years from launch to exit"), "years to exit"
            ),
        )

    def distributions(self) -> dict[str, Distribution]:
        return {
            "tam_usd": self.tam_usd,
            "peak_share": self.peak_share,
            "gross_margin": self.gross_margin,
            "ramp_years": self.ramp_years,
            "exit_revenue_multiple": self.exit_revenue_multiple,
            "years_to_exit": self.years_to_exit,
        }


@dataclass
class DealTerms:
    """The investor's position, as distinct from the project's economics.

    A project can be a fine investment and a terrible one at the same time,
    depending on entry price and dilution. Keeping deal terms separate from the
    market model makes that distinction visible: the engine reports project NPV
    and investor MOIC side by side, and they routinely disagree.
    """

    check_usd: float
    entry_ownership: float
    # Valuation growth between financing rounds, as a multiple of the previous
    # post-money. Dilution is NOT a fixed percentage: it is derived from how much
    # capital each stage actually needs against the valuation it can raise at.
    # This matters more than any other deal term. A programme needing $700M of
    # development capital dilutes an early investor to near-nothing however
    # well the science goes, and a fixed-percentage dilution model hides that
    # completely -- it will happily report a 90x return on a $5M seed cheque
    # into a company that consumed three quarters of a billion dollars.
    valuation_step_up: Distribution
    # Annual rate used to discount project cash flows.
    #
    # IMPORTANT: this should be LOWER than a venture hurdle rate. A 25-30% VC
    # target return is priced to compensate for programme failure, and this model
    # already simulates that failure explicitly, draw by draw. Discounting the
    # surviving cash flows at a venture rate on top of that charges for the same
    # risk twice, and over a 10-15 year development path the double charge is
    # enough to turn genuinely attractive programmes negative. What belongs here
    # is the time value of money plus undiversifiable risk -- a cost of capital,
    # not a hurdle rate.
    discount_rate: Distribution
    provenance: Provenance

    def __post_init__(self) -> None:
        if not 0.0 < self.entry_ownership <= 1.0:
            raise ValueError("entry_ownership must be a fraction in (0, 1]")
        if self.check_usd <= 0:
            raise ValueError("check_usd must be positive")

    @classmethod
    def placeholder(
        cls, check_usd: float = 5.0e6, entry_ownership: float = 0.15
    ) -> "DealTerms":
        note = assumed(
            "placeholder deal terms: a $5M entry for 15%, valuations stepping up "
            "roughly 1.8x between rounds, and a 12% discount rate. The discount "
            "rate is a cost of capital rather than a venture hurdle rate, "
            "because programme attrition is simulated explicitly and must not be "
            "charged for twice. Replace with the actual terms on the table.",
            needs_review=True,
        )
        return cls(
            check_usd=check_usd,
            entry_ownership=entry_ownership,
            valuation_step_up=PERT(1.2, 1.8, 3.0, note, "valuation step-up per round"),
            discount_rate=PERT(0.08, 0.12, 0.18, note, "discount rate"),
            provenance=note,
        )

    @property
    def entry_post_money(self) -> float:
        """Implied post-money valuation at entry."""
        return self.check_usd / self.entry_ownership

    def distributions(self) -> dict[str, Distribution]:
        return {
            "valuation_step_up": self.valuation_step_up,
            "discount_rate": self.discount_rate,
        }
