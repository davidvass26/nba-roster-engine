"""CBA salary-matching rules for trades.

Scope of THIS module
--------------------
Salary matching only: given a team's post-trade salary band and the
salary it sends out, how much salary may it take back? This is the part
of Stage 1 that depends only on base salaries, which our contract source
provides cleanly. Hard-cap TRIGGERS (which apron a move locks a team
into) also live here because they share the band logic, but the parts
that need data we do not yet have -- exact apron payroll including likely
incentives -- are surfaced as uncertainty rather than asserted.

Verified rule set (2023 CBA, fully phased in from 2024-25 onward)
-----------------------------------------------------------------
The salary-matching bands are FIXED DOLLAR FIGURES in the CBA. They do
not scale with the cap, so 2026-27 uses the same bands as 2024-25.

Team BELOW the first apron after the trade, sending out salary S:
    S <= $7,500,000            -> take back up to 200% of S + $250,000
    $7,500,001 <= S <= $29,000,000 -> take back up to S + $7,500,000
    S > $29,000,000            -> take back up to 125% of S + $250,000

Team ABOVE the first apron (either apron) after the trade:
    take back up to 100% of S  (no aggregation benefit; a hard cap applies)

Sources (retrieved 2026-07-27):
    nba.com/news/nba-salary-cap-set-2024-25-season   (apron figures)
    hoopsrumors.com "Running List Of Changes In NBA's New CBA"  (bands)
    hoopsrumors.com "What Each NBA Team Can, Can't Do On The Trade Market"
These match the CBA's transition schedule: the 200/plus-7.5/125 bands for
below-apron teams and the flat 100% ceiling for over-apron teams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from nbare.config import LeagueYear
from nbare.domain.models import CapSheet, TradeProposal
from nbare.domain.money import Money, pct_of

# Fixed CBA band thresholds (dollars). Not cap-relative.
BAND_1_CEILING = 7_500_000     # up to here: 200% + 250K
BAND_2_CEILING = 29_000_000    # up to here: S + 7.5M
MATCHING_ALLOWANCE = 250_000   # the "+$250K" padding (was $100K pre-2023)
BAND_2_FLAT_ADD = 7_500_000    # the "+$7.5M" in the middle band


class Severity(str, Enum):
    LEGAL = "legal"
    ILLEGAL = "illegal"
    # We cannot decide because a needed fact (incentives, dead-money
    # attribution, option type) is missing. This is a first-class outcome,
    # not an error -- an honest "I can't certify this" beats a wrong yes.
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The result of checking one team's side of a trade.

    `max_incoming` is the largest salary the team may legally absorb for
    its outgoing package. `reason` always cites the specific band or rule
    so the output is auditable, never a bare boolean.
    """

    team: str
    severity: Severity
    max_incoming: Money
    actual_incoming: Money
    outgoing: Money
    band: str
    reason: str
    rule: str
    caveats: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.severity is Severity.LEGAL


def matching_ceiling(outgoing: int, *, below_first_apron: bool) -> tuple[int, str, str]:
    """Maximum salary a team may take back for `outgoing`.

    Returns (ceiling_dollars, band_label, cba_reference).

    Over-apron teams get a flat 100% with no padding. Below-apron teams
    get the three-band schedule. The bands are evaluated on the OUTGOING
    salary, and the boundaries are inclusive on the low side per the CBA's
    "up to" / "between" language.
    """
    if not below_first_apron:
        return outgoing, "over_apron_100pct", "CBA VII(k): over-apron 100% match"

    if outgoing <= BAND_1_CEILING:
        ceiling = pct_of(outgoing, "200") + MATCHING_ALLOWANCE
        return ceiling, "band1_200pct+250k", "CBA VII(k)(1): S<=$7.5M"
    if outgoing <= BAND_2_CEILING:
        ceiling = outgoing + BAND_2_FLAT_ADD
        return ceiling, "band2_S+7.5M", "CBA VII(k)(2): $7.5M<S<=$29M"
    ceiling = pct_of(outgoing, "125") + MATCHING_ALLOWANCE
    return ceiling, "band3_125pct+250k", "CBA VII(k)(3): S>$29M"


def _post_trade_apron_payroll(
    sheet: CapSheet, proposal: TradeProposal, ly: LeagueYear
) -> tuple[int, bool, tuple[str, ...]]:
    """Team apron payroll AFTER the trade, and whether we can trust it.

    Band selection depends on where the team sits relative to the first
    apron *after* absorbing the trade. We approximate by swapping outgoing
    cap hits for incoming cap hits. Two honesty problems:
      1. incentives are usually unobserved (apron_payroll is a lower bound)
      2. the incoming players' likely incentives are equally unobserved
    Both are folded into the returned `certain` flag and notes.
    """
    outgoing = proposal.outgoing_for(sheet.team)
    incoming = proposal.incoming_for(sheet.team)
    delta = sum(int(p.cap_hit) for p in incoming) - sum(
        int(p.cap_hit) for p in outgoing
    )
    post = int(sheet.apron_payroll) + delta

    notes: list[str] = list(sheet.uncertainty_notes)
    certain = sheet.certain
    if any(int(p.incentives_likely) == 0 for p in incoming):
        # We can't prove incentives are zero; if the team is near a line
        # this could flip the band.
        margin = abs(post - ly.first_apron)
        if margin < 5_000_000:
            certain = False
            notes.append(
                f"within ${margin:,} of the first apron and incoming likely "
                "incentives are unobserved; band may be wrong"
            )
    return post, certain, tuple(notes)


def check_team_matching(
    sheet: CapSheet,
    proposal: TradeProposal,
    ly: LeagueYear,
) -> Verdict:
    """Salary-matching legality for one team's side of a trade."""
    outgoing_players = proposal.outgoing_for(sheet.team)
    incoming_players = proposal.incoming_for(sheet.team)

    outgoing = Money(sum(int(p.cap_hit) for p in outgoing_players))
    incoming = Money(sum(int(p.cap_hit) for p in incoming_players))

    post_apron, certain, notes = _post_trade_apron_payroll(sheet, proposal, ly)
    below_first = post_apron < ly.first_apron

    ceiling, band, rule = matching_ceiling(
        int(outgoing), below_first_apron=below_first
    )
    ceiling_m = Money(ceiling)

    # A team under the cap can simply absorb salary into cap room; matching
    # only binds teams that are over the cap. We do not model cap room yet
    # (needs cap holds we do not have), so if the team is clearly under the
    # cap post-trade we flag indeterminate rather than assert a ceiling.
    if post_apron < int(ly.salary_cap):
        return Verdict(
            team=sheet.team,
            severity=Severity.INDETERMINATE,
            max_incoming=ceiling_m,
            actual_incoming=incoming,
            outgoing=outgoing,
            band=band,
            reason=(
                "team appears under the salary cap post-trade; cap-room "
                "absorption is not yet modeled (needs cap holds)"
            ),
            rule="cap-room path not modeled",
            caveats=notes,
        )

    severity = Severity.LEGAL if int(incoming) <= ceiling else Severity.ILLEGAL
    if not certain and severity is Severity.LEGAL:
        # Legal on the numbers we can see, but a hidden incentive could push
        # the team over the apron and drop the ceiling. Downgrade.
        severity = Severity.INDETERMINATE

    if severity is Severity.LEGAL:
        reason = (
            f"outgoing ${int(outgoing):,} permits up to ${ceiling:,}; "
            f"incoming ${int(incoming):,} fits"
        )
    elif severity is Severity.ILLEGAL:
        reason = (
            f"incoming ${int(incoming):,} exceeds the ${ceiling:,} ceiling "
            f"for ${int(outgoing):,} outgoing by "
            f"${int(incoming) - ceiling:,}"
        )
    else:
        reason = (
            f"would be legal at ${int(incoming):,} <= ${ceiling:,}, but the "
            "apron band is uncertain (see caveats)"
        )

    return Verdict(
        team=sheet.team,
        severity=severity,
        max_incoming=ceiling_m,
        actual_incoming=incoming,
        outgoing=outgoing,
        band=band,
        reason=reason,
        rule=rule,
        caveats=notes,
    )


@dataclass(frozen=True, slots=True)
class TradeReport:
    """Every team's verdict plus an overall roll-up."""

    verdicts: tuple[Verdict, ...]

    @property
    def severity(self) -> Severity:
        sevs = {v.severity for v in self.verdicts}
        if Severity.ILLEGAL in sevs:
            return Severity.ILLEGAL
        if Severity.INDETERMINATE in sevs:
            return Severity.INDETERMINATE
        return Severity.LEGAL

    @property
    def legal(self) -> bool:
        return self.severity is Severity.LEGAL


def check_trade(
    sheets: dict[str, CapSheet],
    proposal: TradeProposal,
    ly: LeagueYear,
) -> TradeReport:
    """Check salary matching for all teams in a trade.

    A trade is legal on salary-matching grounds only if EVERY team's side
    is legal. Missing a team's cap sheet yields INDETERMINATE for that
    team, never a silent pass.
    """
    verdicts: list[Verdict] = []
    for team in sorted(proposal.teams()):
        sheet = sheets.get(team)
        if sheet is None:
            verdicts.append(
                Verdict(
                    team=team,
                    severity=Severity.INDETERMINATE,
                    max_incoming=Money(0),
                    actual_incoming=Money(
                        sum(int(p.cap_hit) for p in proposal.incoming_for(team))
                    ),
                    outgoing=Money(
                        sum(int(p.cap_hit) for p in proposal.outgoing_for(team))
                    ),
                    band="n/a",
                    reason=f"no cap sheet supplied for {team}",
                    rule="missing input",
                )
            )
            continue
        verdicts.append(check_team_matching(sheet, proposal, ly))
    return TradeReport(verdicts=tuple(verdicts))