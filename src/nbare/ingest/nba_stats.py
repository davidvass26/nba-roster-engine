"""Ingestion from stats.nba.com into the staging layer.

Each function is (a) idempotent, (b) resumable, and (c) reads through
the disk cache, so a killed backfill can be restarted with no loss.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

import polars as pl

from nbare.crosswalk.build import normalize_name
from nbare.ingest.client import NBAStatsCache, fetch

log = logging.getLogger(__name__)


# --- payload -> DataFrame ------------------------------------------------

def result_set_to_df(payload: dict[str, Any], name: str | None = None) -> pl.DataFrame:
    """Convert an nba.com resultSets payload into a Polars DataFrame.

    nba.com returns column-headers-plus-row-arrays rather than records,
    and the header names are inconsistently cased across endpoints, so
    we normalize to snake_case here and only here.
    """
    sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(sets, dict):
        sets = [sets]
    if not sets:
        return pl.DataFrame()

    chosen = None
    if name is not None:
        for rs in sets:
            if rs.get("name") == name:
                chosen = rs
                break
        if chosen is None:
            raise KeyError(f"result set {name!r} not in {[s.get('name') for s in sets]}")
    else:
        chosen = sets[0]

    headers = [_snake(h) for h in chosen.get("headers", [])]
    rows = chosen.get("rowSet", [])
    if not headers:
        return pl.DataFrame()
    return pl.DataFrame(
        rows, schema=headers, orient="row", infer_schema_length=None
    )


def _snake(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.replace(" ", "_").replace("-", "_").lower()


def parse_clock(clock: str | None, period: int) -> float | None:
    """nba.com v3 clock is ISO-8601 duration: 'PT11M23.00S'.

    Returns seconds remaining in the period. Overtime periods are 5:00,
    regulation 12:00 -- callers converting to game-elapsed time must
    account for that, which is why we return period-relative here.
    """
    if not clock:
        return None
    m = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", clock)
    if m:
        mins = int(m.group(1) or 0)
        secs = float(m.group(2) or 0.0)
        return mins * 60 + secs
    # Legacy format 'MM:SS'
    if ":" in clock:
        mm, ss = clock.split(":")
        return int(mm) * 60 + float(ss)
    return None


def parse_minutes_to_seconds(minutes: str | None) -> int | None:
    """Parse a box-score 'minutes played' field to integer seconds.

    nba.com box scores report this as 'MM:SS' (e.g. '34:12') on some
    endpoints and as an ISO-8601 duration ('PT34M12.00S') on others -- the
    same PT-prefixed format `parse_clock` handles for the play-by-play
    clock, so the regex is deliberately the same shape.

    `validate_minutes` compares reconstructed stint-time against this
    field; a silent misparse here makes the minutes gate meaningless
    (CLAUDE.md gotcha: reversed-substitution bugs hide behind a broken
    gate). So unrecognized formats return None (honest "don't know")
    rather than guessing -- EXCEPT an explicitly empty string, which is
    nba.com's own convention for a listed-but-did-not-play player and
    means exactly 0, not "unknown".
    """
    if minutes is None:
        return None
    minutes = minutes.strip()
    if minutes == "":
        return 0

    m = re.match(r"^PT(?:(\d+)M)?(?:([\d.]+)S)?$", minutes)
    if m and (m.group(1) or m.group(2)):
        mins = int(m.group(1) or 0)
        secs = float(m.group(2) or 0.0)
        return int(round(mins * 60 + secs))

    if ":" in minutes:
        mm, ss = minutes.split(":")
        try:
            return int(mm) * 60 + int(round(float(ss)))
        except ValueError:
            return None

    return None


# --- static reference data ----------------------------------------------

def ingest_players(con, cache: NBAStatsCache | None = None) -> int:
    """All players, active and historical."""
    from nba_api.stats.endpoints import commonallplayers

    resp = fetch(
        commonallplayers.CommonAllPlayers,
        {"is_only_current_season": 0, "league_id": "00", "season": "2025-26"},
        cache=cache,
    )
    df = result_set_to_df(resp.payload)
    if df.is_empty():
        return 0

    out = df.select(
        pl.col("person_id").cast(pl.Int64).alias("nba_player_id"),
        pl.col("display_first_last").alias("full_name"),
        pl.col("display_first_last").str.split(" ").list.first().alias("first_name"),
        pl.col("display_first_last").str.split(" ").list.last().alias("last_name"),
        pl.lit(None, dtype=pl.Date).alias("birthdate"),
        pl.lit(None, dtype=pl.Utf8).alias("position"),
        pl.lit(None, dtype=pl.Int16).alias("height_in"),
        pl.lit(None, dtype=pl.Int16).alias("weight_lb"),
        pl.lit(None, dtype=pl.Int16).alias("draft_year"),
        pl.lit(None, dtype=pl.Int16).alias("draft_round"),
        pl.lit(None, dtype=pl.Int16).alias("draft_number"),
        pl.col("from_year").cast(pl.Int16, strict=False).alias("from_year"),
        pl.col("to_year").cast(pl.Int16, strict=False).alias("to_year"),
        (pl.col("rosterstatus").cast(pl.Int32, strict=False) == 1).alias("is_active"),
    ).unique(subset=["nba_player_id"])

    _upsert(con, "stg.player", out, key="nba_player_id")
    return out.height


def ingest_teams(con, cache: NBAStatsCache | None = None) -> int:
    from nba_api.stats.static import teams as static_teams

    rows = static_teams.get_teams()
    df = pl.DataFrame(rows).select(
        pl.col("id").cast(pl.Int64).alias("nba_team_id"),
        pl.col("abbreviation"),
        pl.col("nickname"),
        pl.col("city"),
        pl.col("full_name"),
    )
    _upsert(con, "stg.team", df, key="nba_team_id")
    return df.height


def ingest_season_games(
    con, season: str, season_type: str = "Regular Season",
    cache: NBAStatsCache | None = None,
) -> int:
    """Game index for a season. Source of game_ids for the PBP backfill."""
    from nba_api.stats.endpoints import leaguegamefinder

    resp = fetch(
        leaguegamefinder.LeagueGameFinder,
        {
            "season_nullable": season,
            "season_type_nullable": season_type,
            "league_id_nullable": "00",
        },
        cache=cache,
    )
    df = result_set_to_df(resp.payload)
    if df.is_empty():
        return 0

    # LeagueGameFinder returns one row per team per game; pivot to one
    # row per game using the '@' vs 'vs.' matchup convention.
    df = df.with_columns(
        pl.col("matchup").str.contains("@").alias("is_away")
    )
    home = df.filter(~pl.col("is_away"))
    away = df.filter(pl.col("is_away"))
    games = home.join(away, on="game_id", how="inner", suffix="_away").select(
        pl.col("game_id"),
        pl.lit(season).alias("season"),
        pl.lit(season_type).alias("season_type"),
        pl.col("game_date").str.to_date("%Y-%m-%d", strict=False).alias("game_date"),
        pl.col("team_id").cast(pl.Int64).alias("home_team_id"),
        pl.col("team_id_away").cast(pl.Int64).alias("away_team_id"),
        pl.col("pts").cast(pl.Int16, strict=False).alias("home_pts"),
        pl.col("pts_away").cast(pl.Int16, strict=False).alias("away_pts"),
    )
    _upsert(con, "stg.game", games, key="game_id")
    return games.height


# "SUB: <incoming> FOR <outgoing>" -- confirmed against real 2025-26 games
# (0 misses across 5 games / ~300 substitutions). The outgoing name here is
# redundant with the action's own playerName field (checked as a sanity
# assertion below); the incoming name is the ONLY place its identity
# appears at all.
_SUB_DESCRIPTION_RE = re.compile(r"^SUB: (.+) FOR (.+)$")


def _is_sub_action(event_type: str | None) -> bool:
    """Same substring match as `rapm.stints._is_sub`, duplicated (not
    imported) deliberately: ingest is a lower layer than rapm and should
    not depend on it. Keep these two in sync if nba.com's label changes."""
    return bool(event_type) and "substitution" in event_type.lower()


# Action types confirmed on real games where personId does NOT reliably
# identify a player -- see `_has_reliable_player_id`. Kept even though the
# team_id=0 check below independently catches these too (verified: all
# 1,089 Instant Replay rows across a 400-game audit have team_id=0) --
# defense in depth, and it documents a real discovered case.
_UNRELIABLE_PERSON_ID_ACTION_TYPES = frozenset({"instant replay"})

# The 30 real NBA franchise ids, confirmed against stg.team. A stable,
# long-standing id block (not a guess or an invented fact -- pulled
# directly from ingested team data), used as a second, independent guard
# against a team id leaking into a player slot via some future action type
# we have not seen yet.
_NBA_TEAM_ID_MIN = 1_610_612_737
_NBA_TEAM_ID_MAX = 1_610_612_766


def _has_reliable_player_id(action: dict[str, Any]) -> bool:
    """True if this action's personId can be trusted as an actual player.

    personId is NOT exclusively a player id in v3 -- nba.com reuses it as
    a general "subject of this event" slot. THREE failure modes were
    found by auditing every player1_id/player2_id that landed in
    stg.pbp_event across a 400-game real backfill against stg.player (the
    real player list):

    1. Team-level actions (Timeout, team Rebound, team Turnover) put the
       TEAM's id in personId, always paired with an EMPTY playerName AND
       the row's own teamId field left at 0.
    2. Instant Replay review actions DO carry a non-empty playerName (the
       player under review) but personId is some other small, unrelated
       code -- confirmed by checking that named player's real id
       elsewhere in the same game and finding no match (e.g. "Christie"
       tagged with personId 57 in one game's replay action, never
       appearing under id 57 anywhere else). Also always teamId=0.
    3. A HEAD COACH's technical foul ("T.FOUL") carries the coach's own
       name and an internal id in personId/playerName -- structurally
       identical to a player action (non-empty playerName, actionType
       "Foul") except coaches are not players and will never be in
       stg.player. Found via the audit: personId 2794 ("Bickerstaff") and
       others under actionType "Foul". Distinguished from a genuine
       player technical the same way as (1)/(2): teamId is 0 (confirmed
       against real player technicals in the same games, which have a
       real non-zero teamId).

    All three share one signal: the row's own `teamId` field is 0/absent.
    Requiring it to be truthy is therefore the general guard (confirmed:
    every row in the 400-game audit where player1_id was populated but
    teamId was null was one of these three cases, zero exceptions). The
    playerName check and the explicit action-type exclusion are kept as
    belt-and-suspenders, not the load-bearing check.

    A second, independent guard checks personId itself against the known
    NBA franchise id range -- in case some future, not-yet-seen action
    type leaks a team id through a row that DOES have a nonzero teamId.

    Without these guards, these ids leak into player1_id and
    `_infer_period_openers` treats the team/review-code/coach as a
    phantom player on the floor for a whole period or game -- this is
    what caused the very first real check-minutes run to fail with
    reconstructed errors of exactly 720s/2880s (one period / one full
    game) for ids that were never real players.
    """
    if not action.get("playerName"):
        return False
    if not action.get("teamId"):
        return False
    action_type = (action.get("actionType") or "").lower()
    if action_type in _UNRELIABLE_PERSON_ID_ACTION_TYPES:
        return False
    person_id = action.get("personId")
    if person_id and _NBA_TEAM_ID_MIN <= person_id <= _NBA_TEAM_ID_MAX:
        return False
    return True


def _names_consistent(description_name: str, player_name: str) -> bool:
    """True if `player_name` (the structured field) is a subset of the
    words in `description_name` (the free-text sub name).

    Confirmed on real data: when two teammates share a surname (e.g. two
    Antetokounmpo brothers on the same roster), nba.com's description text
    disambiguates with a leading initial ("G. Antetokounmpo") while the
    structured `playerName` field stays unqualified ("Antetokounmpo") --
    that is expected and must NOT be flagged. A genuine mismatch (the
    Yang/Hansen case: completely different words) should still be flagged.
    Word-subset, not exact-equality, is what tells these apart.
    """
    pn_words = set(normalize_name(player_name).split())
    desc_words = set(normalize_name(description_name).split())
    return bool(pn_words) and pn_words <= desc_words


def _resolve_incoming_id(
    name_to_id: dict[tuple[int, str], int], team_id: int, in_name: str
) -> int | None:
    """Look up an incoming sub's player id by name, tolerating the same
    qualified-vs-unqualified mismatch `_names_consistent` checks for.

    Real bug found by auditing a 400-game backfill: a player can be named
    with a disambiguating initial in a SUBSTITUTION description ("A.
    Johnson", to distinguish him from another Johnson on the roster) while
    his own structured `playerName` -- and therefore his `name_to_id` key,
    learned from his OTHER actions -- stays unqualified ("Johnson"). An
    exact-string lookup on the qualified form misses him entirely, even
    though he is genuinely registered under the unqualified one. This
    surfaced specifically for a player who is NEVER the outgoing side of a
    substitution in that game (the Antetokounmpo-brothers case is caught
    already because the alias-learning loop registers a player's qualified
    name from HIS OWN outgoing sub text -- but that only fires if he ever
    IS the outgoing side, which this player never was).

    First try an exact match (fast path, correct for the overwhelming
    majority of names). If that misses, fall back to a word-subset scan
    over this team's registered names -- the same relationship
    `_names_consistent` already validates the other direction for the
    sanity-check warning.
    """
    exact = name_to_id.get((team_id, normalize_name(in_name)))
    if exact is not None:
        return exact

    in_words = set(normalize_name(in_name).split())
    for (t, registered_name), pid in name_to_id.items():
        if t != team_id:
            continue
        if set(registered_name.split()) <= in_words:
            return pid
    return None


def _safe_int(value: Any) -> int | None:
    """nba.com v3 scores are strings, and '' (not present yet) is common."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_pbp_payload(payload: dict[str, Any], game_id: str) -> pl.DataFrame:
    """Parse a PlayByPlayV3 payload into stg.pbp_event rows.

    Do NOT route this through `result_set_to_df` -- like box scores, v3
    play-by-play does not use the resultSets/rowSet shape. Confirmed by
    reading nba_api's own `NBAStatsPlayByPlayParserV3`
    (nba_api/stats/endpoints/_parsers/playbyplayv3.py) and a real fetched
    game: the raw payload is `{"game": {"gameId": ..., "actions": [...]}}`,
    one dict per action with a SINGLE `personId` field.

    The load-bearing complication: substitutions
    -----------------------------------------------
    A substitution action's own `personId`/`playerName` identify the
    OUTGOING player ONLY (confirmed: cross-checked the description's "FOR
    <name>" clause against `playerName` on every sub in 5 real games --
    always matches). The INCOMING player has no structured id anywhere on
    that action; their name appears only in free text ("SUB: Eason FOR
    Smith Jr."). nba_api's own parser does not resolve this either -- it
    only extracts the single `personId` field.

    We resolve it with a two-pass, self-contained lookup: first scan every
    action in the game to build (team_id, normalized_name) -> person_id
    from whichever OTHER action first reveals that player's id (a shot, a
    rebound, a foul, a later sub) -- exactly the same "recover identity
    from indirect evidence" approach `stints._infer_period_openers` already
    uses for period openers. Names are matched via
    `crosswalk.build.normalize_name` (NOT a fresh ad hoc normalizer) because
    nba.com's free-text descriptions strip diacritics ("Doncic") while the
    structured `playerName` field keeps them ("Dončić") -- confirmed on a
    real game where a naive exact-string match missed a resolvable player
    for exactly this reason.

    Some incoming players are genuinely unresolvable this way: a deep-bench
    player subbed in during garbage time who never touches the ball, is
    never fouled, and is never subbed back out before the period ends
    leaves no OTHER action anywhere to recover their id from. Confirmed
    this happens in real data (~1-2% of subs across 5 real games). Per
    CLAUDE.md principle #1, we do not guess in that case -- `player2_id`
    stays None and it is logged, not silently dropped or fabricated. The
    minutes gate will surface the resulting short lineup; that is correct,
    not a bug to paper over.
    """
    game = payload.get("game")
    if not game:
        log.warning(
            "pbp payload for %s has no 'game' key (got keys: %s) -- skipping",
            game_id, list(payload.keys()),
        )
        return pl.DataFrame()

    actions = game.get("actions") or []
    if not actions:
        return pl.DataFrame()

    name_to_id: dict[tuple[int, str], int] = {}
    for a in actions:
        pid, team_id = a.get("personId"), a.get("teamId")
        if pid and team_id and _has_reliable_player_id(a):
            name_to_id[(team_id, normalize_name(a.get("playerName") or ""))] = pid

        # Alias discovery: a small number of real players (confirmed: a
        # Chinese-name-order rookie whose structured `playerName` is his
        # family name "Yang" but every free-text description -- fouls,
        # rebounds, subs -- calls him "Hansen") have a DIFFERENT name in
        # descriptions than in `playerName`. A substitution's own OUT-side
        # description name is directly paired with that row's own
        # structured personId, so it is not a guess -- register it as an
        # alias so this player also resolves when named as an INCOMING
        # player elsewhere in the game.
        if pid and team_id and _is_sub_action(a.get("actionType")):
            m = _SUB_DESCRIPTION_RE.match(a.get("description") or "")
            if m:
                name_to_id.setdefault((team_id, normalize_name(m.group(2))), pid)

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for a in actions:
        event_type = a.get("actionType")
        team_id = a.get("teamId") or None
        out_id = a.get("personId") if _has_reliable_player_id(a) else None
        in_id = None

        if _is_sub_action(event_type):
            m = _SUB_DESCRIPTION_RE.match(a.get("description") or "")
            if m:
                in_name, matched_out_name = m.group(1), m.group(2)
                if out_id and not _names_consistent(
                    matched_out_name, a.get("playerName") or ""
                ):
                    log.warning(
                        "%s action %s: sub description outgoing name %r doesn't "
                        "match playerName %r -- attribution may be wrong",
                        game_id, a.get("actionNumber"), matched_out_name,
                        a.get("playerName"),
                    )
                in_id = _resolve_incoming_id(name_to_id, team_id, in_name)
                if in_id is None:
                    unresolved += 1

        rows.append({
            "game_id": game_id,
            "event_num": a.get("actionNumber"),
            "period": a.get("period"),
            "clock_seconds_left": parse_clock(a.get("clock"), a.get("period") or 0),
            "event_type": event_type,
            "event_action_type": a.get("subType") or None,
            "description": a.get("description"),
            "team_id": team_id,
            "player1_id": out_id,
            "player2_id": in_id,
            "player3_id": None,
            "home_score": _safe_int(a.get("scoreHome")),
            "away_score": _safe_int(a.get("scoreAway")),
        })

    if unresolved:
        log.warning(
            "%s: %d substitution(s) could not resolve the incoming player id "
            "(never appears elsewhere in this game's actions)",
            game_id, unresolved,
        )

    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Utf8, "event_num": pl.Int32, "period": pl.Int16,
            "clock_seconds_left": pl.Float64, "event_type": pl.Utf8,
            "event_action_type": pl.Utf8, "description": pl.Utf8,
            "team_id": pl.Int64, "player1_id": pl.Int64, "player2_id": pl.Int64,
            "player3_id": pl.Int64, "home_score": pl.Int16, "away_score": pl.Int16,
        },
    ).unique(subset=["game_id", "event_num"], maintain_order=True)


def ingest_pbp(con, game_ids: Iterable[str], cache: NBAStatsCache | None = None) -> int:
    """Play-by-play for a list of games. The long pole of Stage 0."""
    from nba_api.stats.endpoints import playbyplayv3

    total = 0
    for i, gid in enumerate(game_ids, 1):
        try:
            resp = fetch(
                playbyplayv3.PlayByPlayV3,
                {"game_id": gid, "start_period": 0, "end_period": 14},
                cache=cache,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("PBP failed for %s: %s -- skipping, rerun later", gid, exc)
            continue

        out = _parse_pbp_payload(resp.payload, gid)
        if out.is_empty():
            continue

        _upsert(con, "stg.pbp_event", out, key=["game_id", "event_num"])
        total += out.height
        if i % 100 == 0:
            log.info("PBP progress: %d games, %d events", i, total)
    return total


def _parse_box_score_payload(payload: dict[str, Any], game_id: str) -> pl.DataFrame:
    """Parse a BoxScoreTraditionalV3 payload into stg.box_player rows.

    Do NOT route this through `result_set_to_df`. The v3 box score
    endpoints do not use the classic resultSets/rowSet tabular shape --
    confirmed by reading nba_api's own
    `NBAStatsBoxscoreTraditionalParserV3` (nba_api/stats/endpoints/
    _parsers/boxscoretraditionalv3.py), which our `fetch()` wrapper does
    NOT go through (`fetch()` calls `.get_dict()`, the raw, unnormalized
    response -- nba_api's parser is a separate code path we never invoke).
    The real shape is a nested object:

        {"boxScoreTraditional": {
            "gameId": ..., "homeTeam": {"teamId": ..., "players": [...]},
            "awayTeam": {"teamId": ..., "players": [...]},
        }}

    with per-player stats under `player["statistics"]`. Calling
    `result_set_to_df` on this would silently return an empty frame (no
    "resultSets" key exists), the same failure mode already present in
    `ingest_pbp` against real playbyplayv3 payloads -- see CLAUDE.md
    principle #1: an empty result must not be mistaken for "nothing to
    ingest".

    `started` is inferred from `position` being non-empty: nba.com's own
    convention is that only the 5 starters carry a position abbreviation
    ("G"/"F"/"C"/etc.) and bench players get "". This heuristic is
    UNVERIFIED against a real payload as of writing (no confirmed sample
    at the time this was built) -- re-check it against the first real
    fetch and fix here if the real field disagrees.
    """
    box = payload.get("boxScoreTraditional")
    if not box:
        log.warning(
            "box score payload for %s has no boxScoreTraditional key "
            "(got keys: %s) -- skipping", game_id, list(payload.keys()),
        )
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    for team_key in ("homeTeam", "awayTeam"):
        team = box.get(team_key) or {}
        team_id = team.get("teamId")
        for player in team.get("players", []):
            stats = player.get("statistics") or {}
            position = (player.get("position") or "").strip()
            rows.append({
                "game_id": game_id,
                "nba_player_id": player.get("personId"),
                "nba_team_id": team_id,
                "seconds_played": parse_minutes_to_seconds(stats.get("minutes")),
                "pts": stats.get("points"),
                "reb": stats.get("reboundsTotal"),
                "ast": stats.get("assists"),
                "started": bool(position),
            })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Utf8, "nba_player_id": pl.Int64,
            "nba_team_id": pl.Int64, "seconds_played": pl.Int32,
            "pts": pl.Int16, "reb": pl.Int16, "ast": pl.Int16,
            "started": pl.Boolean,
        },
    )


# Full per-player counting stats, beyond the minutes/pts/reb/ast that
# stg.box_player stores. The box-score RAPM prior (rapm/box_features.py)
# needs these; `seconds_played` + team totals are what turn them into
# per-100-possession rates.
BOX_DETAIL_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "nba_player_id": pl.Int64, "nba_team_id": pl.Int64,
    "seconds_played": pl.Int32, "position": pl.Utf8,
    "pts": pl.Int32, "ast": pl.Int32, "tov": pl.Int32,
    "fga": pl.Int32, "fta": pl.Int32,
    "oreb": pl.Int32, "dreb": pl.Int32, "stl": pl.Int32, "blk": pl.Int32,
}


def parse_box_score_detail(payload: dict[str, Any], game_id: str) -> pl.DataFrame:
    """Parse a BoxScoreTraditionalV3 payload into full per-player counting
    stats (see BOX_DETAIL_SCHEMA). Same nested payload shape as
    `_parse_box_score_payload` -- read its docstring for why this must not
    go through `result_set_to_df`.

    `position` is kept raw ("G"/"F"/"C"/... for starters, "" for bench):
    nba.com only labels starters, so it is a partial signal and callers
    must treat "" as UNKNOWN, not as "not a big". A missing counting stat
    stays null rather than becoming 0.
    """
    box = payload.get("boxScoreTraditional")
    if not box:
        return pl.DataFrame(schema=BOX_DETAIL_SCHEMA)

    rows: list[dict[str, Any]] = []
    for team_key in ("homeTeam", "awayTeam"):
        team = box.get(team_key) or {}
        team_id = team.get("teamId")
        for player in team.get("players", []):
            stats = player.get("statistics") or {}
            rows.append({
                "game_id": game_id,
                "nba_player_id": player.get("personId"),
                "nba_team_id": team_id,
                "seconds_played": parse_minutes_to_seconds(stats.get("minutes")),
                "position": (player.get("position") or "").strip(),
                "pts": stats.get("points"),
                "ast": stats.get("assists"),
                "tov": stats.get("turnovers"),
                "fga": stats.get("fieldGoalsAttempted"),
                "fta": stats.get("freeThrowsAttempted"),
                "oreb": stats.get("reboundsOffensive"),
                "dreb": stats.get("reboundsDefensive"),
                "stl": stats.get("steals"),
                "blk": stats.get("blocks"),
            })
    return pl.DataFrame(rows, schema=BOX_DETAIL_SCHEMA)


def box_detail_from_cache(
    game_ids: Iterable[str], cache: NBAStatsCache | None = None
) -> tuple[pl.DataFrame, list[str]]:
    """Full box-score detail for `game_ids`, read from the disk cache ONLY.

    Never fetches: this runs as part of a fit, and a fit must not silently
    turn into a rate-limited nba.com backfill (or fail in a sandbox where
    nba.com is unreachable). Games with no cached box score are returned
    in the second element so the caller can report them honestly instead
    of pretending coverage is complete. Run `make box SEASON=...` to fill
    them.
    """
    from nbare.ingest.client import request_hash

    if cache is None:
        cache = NBAStatsCache()
    frames: list[pl.DataFrame] = []
    missing: list[str] = []
    for gid in game_ids:
        hit = cache.get(request_hash("BoxScoreTraditionalV3", {"game_id": gid}))
        if hit is None:
            missing.append(gid)
            continue
        df = parse_box_score_detail(hit["payload"], gid)
        if df.is_empty():
            missing.append(gid)
            continue
        frames.append(df)
    detail = pl.concat(frames) if frames else pl.DataFrame(schema=BOX_DETAIL_SCHEMA)
    return detail, missing


# --- player bio: per-season age + position group ---------------------------
# Stage 3 projections need every player-season's AGE and every player's
# position GROUP, and stg.player has neither (birthdate/position are empty
# for all rows -- CommonAllPlayers does not carry them). Two league-wide
# endpoints supply them in one request per season each, instead of ~800
# per-player CommonPlayerInfo calls:
#
#   LeagueDashPlayerStats(season)  -> AGE per player for that season.
#     Verified against known birthdates: LeBron 39/40/41 and Curry
#     36/37/38 for 2023-24/2024-25/2025-26, i.e. nba.com's AGE behaves like
#     age at season END (Curry turns a year older in March). The exact
#     convention matters less than that it is applied the same way every
#     season, so consecutive seasons differ by exactly 1 -- what the delta
#     method needs; projection/panel.py checks that per player.
#   PlayerIndex(season)            -> POSITION ("G", "F-C", ...). Past
#     seasons' indexes only list ~140 players, the current one ~580, so
#     the most recent season that lists a player wins.

def parse_player_ages(payload: dict[str, Any]) -> dict[int, int]:
    """LeagueDashPlayerStats payload -> {player_id: age}. A null age is
    left out (unknown), never defaulted."""
    df = result_set_to_df(payload)
    if df.is_empty() or "age" not in df.columns:
        return {}
    return {
        int(pid): int(age)
        for pid, age in df.select("player_id", "age").iter_rows()
        if pid is not None and age is not None
    }


def position_group(position: str | None) -> str | None:
    """nba.com position string -> coarse group for trajectory pooling.

    The FIRST listed position wins ("G-F" -> "G", "F-C" -> "F", "C-F" ->
    "C"): nba.com lists a player's primary position first. Empty/unknown
    -> None, which callers must keep distinct from any real group.

    >>> [position_group(p) for p in ("G", "G-F", "F-C", "C-F", "", None)]
    ['G', 'G', 'F', 'C', None, None]
    """
    if not position:
        return None
    head = position.strip().split("-")[0].upper()
    return head if head in ("G", "F", "C") else None


def parse_player_positions(payload: dict[str, Any]) -> dict[int, str]:
    """PlayerIndex payload -> {player_id: position group}; players with no
    usable position are left out."""
    df = result_set_to_df(payload)
    if df.is_empty() or "position" not in df.columns:
        return {}
    out: dict[int, str] = {}
    for pid, pos in df.select("person_id", "position").iter_rows():
        g = position_group(pos)
        if pid is not None and g is not None:
            out[int(pid)] = g
    return out


def _bio_requests(season: str) -> list[tuple[Any, dict[str, Any]]]:
    from nba_api.stats.endpoints import leaguedashplayerstats, playerindex

    return [
        (leaguedashplayerstats.LeagueDashPlayerStats,
         {"season": season, "season_type_all_star": "Regular Season"}),
        (playerindex.PlayerIndex, {"season": season, "league_id": "00"}),
    ]


def ingest_player_bio(seasons: Iterable[str], cache: NBAStatsCache | None = None) -> int:
    """Fetch (through the cached, rate-limited client) the two bio
    endpoints for each season. Returns the number of responses now cached.
    Resumable for free: already-cached seasons are cache hits."""
    n = 0
    for season in seasons:
        for cls, params in _bio_requests(season):
            fetch(cls, params, cache=cache)
            n += 1
    return n


def player_bio_from_cache(
    seasons: list[str], cache: NBAStatsCache | None = None
) -> tuple[dict[tuple[int, str], int], dict[int, str], list[str]]:
    """(ages keyed by (player_id, season), position group by player_id,
    seasons missing a cached bio response). Cache ONLY -- never fetches;
    run `nbare ingest-bio` first. Positions: the latest season listing a
    player wins (see the note above)."""
    from nbare.ingest.client import request_hash

    if cache is None:
        cache = NBAStatsCache()
    ages: dict[tuple[int, str], int] = {}
    positions: dict[int, str] = {}
    missing: list[str] = []
    for season in sorted(seasons):
        (age_cls, age_params), (idx_cls, idx_params) = _bio_requests(season)
        age_hit = cache.get(request_hash(age_cls.__name__, age_params))
        idx_hit = cache.get(request_hash(idx_cls.__name__, idx_params))
        if age_hit is None or idx_hit is None:
            missing.append(season)
        if age_hit is not None:
            for pid, age in parse_player_ages(age_hit["payload"]).items():
                ages[(pid, season)] = age
        if idx_hit is not None:
            positions.update(parse_player_positions(idx_hit["payload"]))
    return ages, positions, missing


def ingest_box_scores(con, game_ids: Iterable[str], cache: NBAStatsCache | None = None) -> int:
    """Traditional box score for a list of games.

    Mirrors `ingest_pbp`: cached, rate-limited, resumable, and a single
    game's failure is logged and skipped rather than aborting the whole
    backfill. This is what fills `stg.box_player`, which `check-minutes`
    and `rapm.blocks.blocks_from_warehouse` already read but nothing wrote
    to (see docs/box_ingest_spec.md).
    """
    from nba_api.stats.endpoints import boxscoretraditionalv3

    total = 0
    for i, gid in enumerate(game_ids, 1):
        try:
            resp = fetch(
                boxscoretraditionalv3.BoxScoreTraditionalV3,
                {"game_id": gid},
                cache=cache,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Box score failed for %s: %s -- skipping, rerun later", gid, exc)
            continue

        df = _parse_box_score_payload(resp.payload, gid)
        if df.is_empty():
            continue

        _upsert(con, "stg.box_player", df, key=["game_id", "nba_player_id"])
        total += df.height
        if i % 100 == 0:
            log.info("Box score progress: %d games, %d rows", i, total)
    return total


# --- upsert helper -------------------------------------------------------

def _upsert(con, table: str, df: pl.DataFrame, key: str | list[str]) -> None:
    """Delete-then-insert on primary key. DuckDB has no MERGE we can rely
    on across versions, and this is fast enough for our volumes."""
    if df.is_empty():
        return
    keys = [key] if isinstance(key, str) else list(key)
    con.register("_incoming", df.to_arrow())
    pred = " AND ".join(f't."{k}" = i."{k}"' for k in keys)
    con.execute(
        f"DELETE FROM {table} t WHERE EXISTS "
        f"(SELECT 1 FROM _incoming i WHERE {pred})"
    )
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    con.unregister("_incoming")
