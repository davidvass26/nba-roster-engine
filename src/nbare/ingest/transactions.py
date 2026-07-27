"""Apply roster transactions on top of a base contract snapshot.

Why an overlay instead of editing the CSV
------------------------------------------
The contract CSV is a point-in-time snapshot. Real life moves faster than
your exports: a player signs, gets waived, gets bought out, gets traded.
Editing the CSV by hand to reflect those is error-prone (you will forget a
cap hold, or leave a player on two teams) and destroys your ability to say
"what did the league look like on the day I pulled this."

So the base stays immutable and changes live in a small, ordered list of
*transactions* in a YAML file. Applying them produces a new, derived
snapshot. This mirrors the raw/derived split in the rest of the project:
the source is sacred, the overlay is disposable and diffable.

Why transactions and not "set player X's team to Y"
---------------------------------------------------
The CBA treats a signing, a waiver, a buyout, and a trade as different
operations with different cap consequences, and the Stage 1 rule engine
already knows the difference. A find-and-replace would flatten all of that
into "change a string" and quietly produce illegal or nonsensical cap
sheets. A veteran-minimum signing is not the same cap hit as the salary
the player negotiated; a waived non-guaranteed player comes off the books
while a stretched guaranteed one leaves dead money behind. The transaction
types below preserve those distinctions.

Supported transactions
-----------------------
- sign      : a free agent joins a team at a stated cap hit
- waive     : a player leaves a team (dead money handling depends on
              whether the deal was guaranteed)
- trade     : players (and cash) move between teams
- amend     : override a single field on an existing row (escape hatch for
              corrections the other types do not cover)

Each carries an `as_of` date and a free-text `note` so the overlay reads
like a transaction log, not a diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml

# --- transaction types ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sign:
    """A free agent signs with a team.

    `cap_hit` is REQUIRED and is the cap number, which for minimum deals is
    often below the stated contract value (the league reimburses the
    difference on vet minimums). Do not put the headline salary here if it
    differs from the cap hit -- the rule engine reasons on cap hits.
    """

    player: str          # display name (new signees may have no slug yet)
    team: str
    cap_hit: int
    guaranteed: int
    slug: str | None = None
    note: str = ""
    as_of: date | None = None
    kind: Literal["sign"] = "sign"


@dataclass(frozen=True, slots=True)
class Waive:
    """A team waives a player.

    `stretch_dead_money` distinguishes a clean release of a non-guaranteed
    deal (player leaves, nothing remains) from a waived guaranteed contract
    that leaves dead money on the team's books. When True, the player's
    guaranteed amount stays as a dead-money row rather than disappearing.
    """

    slug: str
    team: str
    stretch_dead_money: bool = False
    note: str = ""
    as_of: date | None = None
    kind: Literal["waive"] = "waive"


@dataclass(frozen=True, slots=True)
class TradeLeg:
    slug: str
    to_team: str


@dataclass(frozen=True, slots=True)
class Trade:
    """Players move between teams. `legs` lists each player's destination;
    the source is inferred from the base snapshot."""

    legs: tuple[TradeLeg, ...]
    cash: dict[str, int] = field(default_factory=dict)
    note: str = ""
    as_of: date | None = None
    kind: Literal["trade"] = "trade"


@dataclass(frozen=True, slots=True)
class Amend:
    """Override one field of one existing row. Escape hatch for corrections
    the structured types do not cover (fixing a wrong salary, flipping a
    guarantee). Use sparingly; prefer the specific types above."""

    slug: str
    field: str
    value: Any
    note: str = ""
    as_of: date | None = None
    kind: Literal["amend"] = "amend"


Transaction = Sign | Waive | Trade | Amend


# --- loading -------------------------------------------------------------

def load_transactions(path: Path | str) -> list[Transaction]:
    """Parse a YAML overlay file into typed transactions.

    Format (a list of single-key dicts, applied top to bottom):

        - sign:
            player: LeBron James
            slug: jamesle01
            team: PHI
            cap_hit: 54126380
            guaranteed: 54126380
            note: signed as FA 2026-07-25
        - waive:
            slug: watfotr01
            team: PHI
            note: waived to open a roster spot
        - trade:
            legs:
              - {slug: banede01, to_team: NYK}
              - {slug: bridgmi01, to_team: ORL}
            note: hypothetical Bane/Bridges swap
    """
    raw = yaml.safe_load(Path(path).read_text()) or []
    out: list[Transaction] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"transaction {i} must be a single-key mapping, got {entry!r}"
            )
        (kind, body), = entry.items()
        out.append(_parse_one(kind, body, i))
    return out


def _parse_one(kind: str, body: dict, i: int) -> Transaction:
    body = dict(body)
    as_of = body.pop("as_of", None)
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    if kind == "sign":
        return Sign(as_of=as_of, **body)
    if kind == "waive":
        return Waive(as_of=as_of, **body)
    if kind == "amend":
        return Amend(as_of=as_of, **body)
    if kind == "trade":
        legs = tuple(
            TradeLeg(slug=l["slug"], to_team=l["to_team"]) for l in body.pop("legs")
        )
        return Trade(legs=legs, as_of=as_of, **body)
    raise ValueError(f"transaction {i}: unknown kind {kind!r}")


# --- application ---------------------------------------------------------

@dataclass
class ApplyResult:
    frame: pl.DataFrame
    log: list[str]
    warnings: list[str]


def apply_transactions(
    base: pl.DataFrame,
    transactions: list[Transaction],
    season: str = "2026-27",
) -> ApplyResult:
    """Apply transactions to a contract-year frame, returning a new frame.

    Operates on the exploded contract-year frame (output of
    contracts.to_contract_years), filtered to one season, so cap_hit is a
    single number per player. Pure: the input frame is not mutated.

    Every action is logged; anything suspicious (waiving a player who is
    not on the named team, signing a player who already exists) is recorded
    as a warning rather than silently applied, because a wrong overlay that
    runs clean is worse than one that complains.
    """
    df = base.filter(pl.col("season") == season).clone()
    log: list[str] = []
    warnings: list[str] = []

    for txn in transactions:
        if isinstance(txn, Sign):
            df, w = _apply_sign(df, txn, season)
        elif isinstance(txn, Waive):
            df, w = _apply_waive(df, txn)
        elif isinstance(txn, Trade):
            df, w = _apply_trade(df, txn)
        elif isinstance(txn, Amend):
            df, w = _apply_amend(df, txn)
        else:  # pragma: no cover
            raise TypeError(f"unknown transaction {txn!r}")
        log.append(_describe(txn))
        warnings.extend(w)

    return ApplyResult(frame=df, log=log, warnings=warnings)


def _row(df: pl.DataFrame, slug: str) -> dict | None:
    hit = df.filter(pl.col("bbref_slug") == slug)
    return hit.to_dicts()[0] if hit.height else None


def _apply_sign(df, txn: Sign, season: str) -> tuple[pl.DataFrame, list[str]]:
    w: list[str] = []
    existing = _row(df, txn.slug) if txn.slug else None
    if existing is not None:
        w.append(
            f"sign: {txn.player} ({txn.slug}) already present on "
            f"{existing['team_abbrev']}; replacing with new {txn.team} deal"
        )
        df = df.filter(pl.col("bbref_slug") != txn.slug)
    new = {
        "contract_id": f"override:{txn.slug or txn.player}",
        "season": season,
        "bbref_slug": txn.slug or f"__new__{txn.player.replace(' ', '_').lower()}",
        "player": txn.player,
        "team_abbrev": txn.team,
        "season_index": 1,
        "cap_hit": txn.cap_hit,
        "guaranteed": txn.guaranteed,
        "is_guaranteed": txn.guaranteed >= txn.cap_hit,
        "option_type": None,
        "needs_review": False,
        "review_reason": None,
    }
    add = pl.DataFrame([new], schema=df.schema)
    return pl.concat([df, add], how="vertical"), w


def _apply_waive(df, txn: Waive) -> tuple[pl.DataFrame, list[str]]:
    w: list[str] = []
    row = _row(df, txn.slug)
    if row is None:
        w.append(f"waive: {txn.slug} not found; no-op")
        return df, w
    if row["team_abbrev"] != txn.team:
        w.append(
            f"waive: {txn.slug} is on {row['team_abbrev']}, not {txn.team}; "
            "applying to the team they are actually on"
        )
    df = df.filter(pl.col("bbref_slug") != txn.slug)
    if txn.stretch_dead_money and row["guaranteed"] > 0:
        dead = dict(row)
        dead["cap_hit"] = row["guaranteed"]
        dead["needs_review"] = True
        dead["review_reason"] = "waived_dead_money"
        dead["contract_id"] = f"dead:{txn.slug}"
        df = pl.concat([df, pl.DataFrame([dead], schema=df.schema)], how="vertical")
    return df, w


def _apply_trade(df, txn: Trade) -> tuple[pl.DataFrame, list[str]]:
    w: list[str] = []
    updates: dict[str, str] = {}
    for leg in txn.legs:
        row = _row(df, leg.slug)
        if row is None:
            w.append(f"trade: {leg.slug} not found; leg skipped")
            continue
        updates[leg.slug] = leg.to_team
    if not updates:
        return df, w
    expr = pl.col("team_abbrev")
    for slug, team in updates.items():
        expr = pl.when(pl.col("bbref_slug") == slug).then(pl.lit(team)).otherwise(expr)
    return df.with_columns(expr.alias("team_abbrev")), w


def _apply_amend(df, txn: Amend) -> tuple[pl.DataFrame, list[str]]:
    w: list[str] = []
    if _row(df, txn.slug) is None:
        w.append(f"amend: {txn.slug} not found; no-op")
        return df, w
    if txn.field not in df.columns:
        raise ValueError(f"amend: unknown field {txn.field!r}")
    expr = (
        pl.when(pl.col("bbref_slug") == txn.slug)
        .then(pl.lit(txn.value))
        .otherwise(pl.col(txn.field))
        .alias(txn.field)
    )
    return df.with_columns(expr), w


def _describe(txn: Transaction) -> str:
    d = txn.as_of.isoformat() if txn.as_of else "----------"
    if isinstance(txn, Sign):
        return f"[{d}] SIGN  {txn.player} -> {txn.team} @ ${txn.cap_hit:,}  {txn.note}"
    if isinstance(txn, Waive):
        tag = " (+dead money)" if txn.stretch_dead_money else ""
        return f"[{d}] WAIVE {txn.slug} from {txn.team}{tag}  {txn.note}"
    if isinstance(txn, Trade):
        moves = ", ".join(f"{l.slug}->{l.to_team}" for l in txn.legs)
        return f"[{d}] TRADE {moves}  {txn.note}"
    if isinstance(txn, Amend):
        return f"[{d}] AMEND {txn.slug}.{txn.field}={txn.value}  {txn.note}"
    return str(txn)