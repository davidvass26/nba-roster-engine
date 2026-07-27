"""Domain models for the Stage 1 rule engine.

These are the typed inputs the CBA rules operate on. They are deliberately
minimal and honest: a field is present only when the current data sources
can actually populate it. Where a fact is unknowable from what we have
(likely incentives, option types), it is Optional and defaults to a value
that makes the rule engine *refuse to certify* rather than silently assume
the favorable case.

Nothing here does cap math. Money arithmetic lives in domain.money; the
rules live in cba.matching. Keeping the data separate from the rules is
what lets the same TradeProposal be replayed against any LeagueYear for
backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nbare.domain.money import Money


class OptionType(str, Enum):
    """Contract option on a season. NON_GUARANTEED and PLAYER are different
    cap facts even when the dollar figure matches, so they are distinct."""

    PLAYER = "player"
    TEAM = "team"
    EARLY_TERMINATION = "eto"
    NON_GUARANTEED = "non_guaranteed"


@dataclass(frozen=True, slots=True)
class PlayerSalary:
    """One player's cap situation for a single season, as far as we can
    know it from present data.

    `incoming_incentives_likely` is the honesty valve. Apron payroll
    legally includes likely incentives, and our contract source does not
    carry them. Default 0 means "we could not observe any", and any cap
    sheet built from such players is flagged as a lower bound rather than
    presented as exact.
    """

    player_id: str
    name: str
    cap_hit: Money
    guaranteed: Money
    is_dead_money: bool = False
    # None == unknown from this source (NOT the same as "no option").
    option_type: OptionType | None = None
    incentives_likely: Money = field(default_factory=lambda: Money(0))
    has_trade_bonus: bool = False
    no_trade_clause: bool = False

    def __post_init__(self) -> None:
        if self.cap_hit < 0:
            raise ValueError(f"negative cap hit for {self.name}")


@dataclass(frozen=True, slots=True)
class CapSheet:
    """A team's salary state for one season.

    `certain` is False when any constituent salary carries an unknown that
    could move the team across an apron line -- unobserved incentives, a
    dead-money row we could not attribute, an unknown option. The rule
    engine reads this flag and downgrades any conclusion that depends on
    the exact apron band from a verdict to an advisory.
    """

    team: str
    season: str
    salaries: tuple[PlayerSalary, ...]
    trade_exceptions: tuple[Money, ...] = ()
    certain: bool = True
    uncertainty_notes: tuple[str, ...] = ()

    @property
    def roster_count(self) -> int:
        return sum(1 for s in self.salaries if not s.is_dead_money)

    @property
    def cap_payroll(self) -> Money:
        """Total salary counted against the cap (includes dead money)."""
        return Money(sum(int(s.cap_hit) for s in self.salaries))

    @property
    def apron_payroll(self) -> Money:
        """Salary counted toward the aprons: cap payroll plus LIKELY
        incentives. With our current sources the incentive term is almost
        always zero, which is exactly why `certain` matters -- this is a
        lower bound on the true apron figure."""
        base = int(self.cap_payroll)
        inc = sum(int(s.incentives_likely) for s in self.salaries)
        return Money(base + inc)


@dataclass(frozen=True, slots=True)
class TradePiece:
    """Salary leaving one team in a trade. A trade is a set of these."""

    player: PlayerSalary
    from_team: str
    to_team: str


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A proposed trade: who sends what to whom, plus cash.

    Supports the two-team case fully and is shaped to extend to N teams.
    Aggregation (combining multiple outgoing salaries to match one
    incoming) is represented naturally by a team having more than one
    outgoing piece -- which is itself an apron-restricted action, so the
    structure carries the information the rules need.
    """

    pieces: tuple[TradePiece, ...]
    cash_sent: dict[str, Money] = field(default_factory=dict)

    def teams(self) -> set[str]:
        t: set[str] = set()
        for p in self.pieces:
            t.add(p.from_team)
            t.add(p.to_team)
        return t

    def outgoing_for(self, team: str) -> tuple[PlayerSalary, ...]:
        return tuple(p.player for p in self.pieces if p.from_team == team)

    def incoming_for(self, team: str) -> tuple[PlayerSalary, ...]:
        return tuple(p.player for p in self.pieces if p.to_team == team)

    def aggregates_salary(self, team: str) -> bool:
        """True if the team combines two or more outgoing salaries. This is
        prohibited above the second apron and is a hard-cap trigger when
        done between the aprons."""
        return len(self.outgoing_for(team)) >= 2