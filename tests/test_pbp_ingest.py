"""Tests for the play-by-play v3 parser (docs/pbp_parse_fix_spec.md).

The substitution in/out attribution is the load-bearing field here --
stint reconstruction reads player1_id (OUT) / player2_id (IN) directly to
build lineups, and CLAUDE.md's own "reversed substitution direction"
gotcha exists because of exactly this kind of bug. So the headline tests
here assert on that attribution specifically, not just "the DataFrame is
non-empty" -- mirroring what worked for the box-score fix.

Payload shapes below are hand-built to match the REAL structure confirmed
by fetching actual 2025-26 games (see the parser's docstring for what was
verified and how).
"""

from __future__ import annotations

import polars as pl

from nbare.ingest.nba_stats import _parse_pbp_payload


def _action(action_number, period, clock, action_type, person_id=0,
            team_id=0, player_name="", description="", sub_type="",
            score_home="", score_away=""):
    return {
        "actionNumber": action_number, "clock": clock, "period": period,
        "teamId": team_id, "teamTricode": "", "personId": person_id,
        "playerName": player_name, "playerNameI": "", "description": description,
        "actionType": action_type, "subType": sub_type,
        "scoreHome": score_home, "scoreAway": score_away,
    }


def test_parse_pbp_payload_missing_key_returns_empty():
    assert _parse_pbp_payload({}, "0022500001").is_empty()
    assert _parse_pbp_payload({"unrelated": 1}, "0022500001").is_empty()


def test_parse_pbp_payload_empty_actions_returns_empty():
    assert _parse_pbp_payload({"game": {"gameId": "x", "actions": []}}, "x").is_empty()


def test_parse_pbp_payload_shape():
    payload = {"game": {"gameId": "0022500001", "actions": [
        _action(1, 1, "PT12M00.00S", "period", score_home="0", score_away="0"),
    ]}}
    df = _parse_pbp_payload(payload, "0022500001")
    assert df.columns == [
        "game_id", "event_num", "period", "clock_seconds_left", "event_type",
        "event_action_type", "description", "team_id", "player1_id",
        "player2_id", "player3_id", "home_score", "away_score",
    ]
    assert df.height == 1
    row = df.to_dicts()[0]
    assert row["clock_seconds_left"] == 720.0
    assert row["home_score"] == 0
    assert row["away_score"] == 0


# --- the load-bearing field: substitution in/out attribution ---------------

def test_instant_replay_does_not_leak_bogus_id_into_player1_id():
    """Real bug found on real data: Instant Replay review actions DO carry
    a non-empty playerName (the player under review) but personId is some
    other small, unrelated code -- confirmed by checking that named player
    ('Christie') never appears under that id (57) anywhere else in the
    same game. A playerName-only guard does not catch this; it needs an
    explicit action-type exclusion."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(1, 1, "PT07M53.00S", "Instant Replay", person_id=57, team_id=0,
                player_name="Christie", description="Instant Replay1st Period"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    assert df["player1_id"].to_list() == [None]


def test_team_level_events_do_not_leak_team_id_into_player1_id():
    """Real bug found on real data: nba.com puts the TEAM's id in personId
    for team-level actions (Timeout, team Rebound, team Turnover). Real
    payloads confirmed: these rows ALSO leave the row's own teamId field
    at 0 (not the team's real id) -- both personId=team_id AND
    playerName="" AND teamId=0 together, matching an audited real payload.
    Without a guard, that team id leaked into player1_id and
    `_infer_period_openers` treated the TEAM as a phantom player on the
    floor for a whole period/game -- this is what caused the very first
    real check-minutes run to fail with errors of exactly 720s/2880s for
    ids that were actually team ids."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(1, 1, "PT08M00.00S", "Timeout", person_id=1610612744,
                team_id=0, player_name="", description="Warriors Timeout"),
        _action(2, 1, "PT07M00.00S", "Rebound", person_id=1610612747,
                team_id=0, player_name="", description="LAKERS Rebound"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    assert df["player1_id"].to_list() == [None, None]


def test_coach_technical_foul_does_not_leak_into_player1_id():
    """Real bug found by auditing every player1_id/player2_id in a
    400-game real backfill against stg.player: a HEAD COACH's technical
    foul carries the coach's own name/id in personId/playerName,
    structurally identical to a player action (non-empty playerName,
    actionType "Foul") -- but coaches are never in stg.player. Confirmed
    on a real game: J.B. Bickerstaff's (a real NBA head coach) technical
    foul has personId 2794, playerName "Bickerstaff", and (the
    distinguishing signal, confirmed against a real PLAYER technical in
    the same game which has a proper non-zero teamId) teamId 0."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(39, 1, "PT09M51.00S", "Foul", person_id=2794, team_id=0,
                player_name="Bickerstaff", sub_type="Technical",
                description="John-Blair Bickerstaff Foul:T.FOUL (E.Dalen)"),
        # A genuine player technical, for contrast -- must NOT be excluded.
        _action(40, 1, "PT03M00.00S", "Foul", person_id=1628976, team_id=1610612753,
                player_name="Carter Jr.", sub_type="Technical",
                description="Carter Jr. T.FOUL (P3.T4) (E.Dalen)"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    assert df["player1_id"].to_list() == [None, 1628976]


def test_team_id_range_guard_catches_unseen_leak_shape():
    """Defense in depth: even if a future action type leaks a team id
    through a row that DOES have a nonzero teamId (unlike every leak
    class found so far), the personId-against-known-franchise-id-range
    check still catches it."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(1, 1, "PT08M00.00S", "SomeFutureActionType",
                person_id=1610612737, team_id=1610612737,
                player_name="Hawks", description="hypothetical future leak"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    assert df["player1_id"].to_list() == [None]


def test_substitution_out_id_from_own_person_id():
    """The row's own personId/playerName is the OUTGOING player -- confirmed
    against real games (the description's 'FOR <name>' clause always
    matches playerName)."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(10, 1, "PT06M40.00S", "Made Shot", person_id=200, team_id=1,
                player_name="Eason"),  # incoming player does something else
        _action(20, 1, "PT06M15.00S", "Substitution", person_id=100, team_id=1,
                player_name="Smith Jr.", description="SUB: Eason FOR Smith Jr."),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    sub_row = df.filter(pl.col("event_num") == 20).to_dicts()[0]
    assert sub_row["player1_id"] == 100  # OUT: Smith Jr.
    assert sub_row["player2_id"] == 200  # IN: Eason, resolved from action 10


def test_substitution_incoming_id_resolved_from_later_action():
    """The incoming player's identifying action can come AFTER the sub in
    event order -- resolution must not depend on chronological order."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(20, 1, "PT06M15.00S", "Substitution", person_id=100, team_id=1,
                player_name="Smith Jr.", description="SUB: Eason FOR Smith Jr."),
        _action(30, 1, "PT05M00.00S", "Rebound", person_id=200, team_id=1,
                player_name="Eason"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    sub_row = df.filter(pl.col("event_num") == 20).to_dicts()[0]
    assert sub_row["player2_id"] == 200


def test_substitution_resolves_diacritics_mismatch():
    """Real bug found on a real game: playerName carries diacritics
    ('Dončić') but the free-text description is plain ASCII ('Doncic').
    Must resolve via normalize_name, not exact string match."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(5, 1, "PT10M00.00S", "Made Shot", person_id=999, team_id=1,
                player_name="Dončić"),
        _action(20, 1, "PT06M15.00S", "Substitution", person_id=100, team_id=1,
                player_name="Reaves", description="SUB: Doncic FOR Reaves"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    sub_row = df.filter(pl.col("event_num") == 20).to_dicts()[0]
    assert sub_row["player1_id"] == 100
    assert sub_row["player2_id"] == 999


def test_substitution_unresolvable_incoming_id_is_none_not_guessed():
    """A garbage-time sub who never appears elsewhere in the game's actions
    is a genuine, honest gap -- confirmed this happens in real data. Must
    stay None, never fabricated (CLAUDE.md principle #1)."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(20, 1, "PT06M15.00S", "Substitution", person_id=100, team_id=1,
                player_name="Bona", description="SUB: Edwards FOR Bona"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    sub_row = df.filter(pl.col("event_num") == 20).to_dicts()[0]
    assert sub_row["player1_id"] == 100
    assert sub_row["player2_id"] is None


def test_substitution_resolves_same_surname_teammates_correctly(caplog):
    """Real bug risk found on a real game: two brothers (Giannis and
    Thanasis Antetokounmpo) on the same roster, both with structured
    playerName 'Antetokounmpo' (no qualifier). nba.com's description text
    disambiguates them with a leading initial ('G. Antetokounmpo' /
    'T. Antetokounmpo'). Each must resolve to their OWN id, never
    cross-wired to the other, and this qualified-vs-unqualified difference
    must NOT be flagged as a mismatch (it is expected, not an error)."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(10, 1, "PT10M00.00S", "Made Shot", person_id=1, team_id=1,
                player_name="Antetokounmpo", description="G. Antetokounmpo Layup"),
        # Giannis (id 1) subbed OUT -- teaches alias "g antetokounmpo" -> 1.
        _action(20, 1, "PT08M00.00S", "Substitution", person_id=1, team_id=1,
                player_name="Antetokounmpo", description="SUB: Kuzma FOR G. Antetokounmpo"),
        # Thanasis (id 2) subbed OUT -- teaches alias "t antetokounmpo" -> 2.
        _action(25, 1, "PT07M00.00S", "Substitution", person_id=2, team_id=1,
                player_name="Antetokounmpo", description="SUB: Trent FOR T. Antetokounmpo"),
        # Giannis subbed back IN -- must resolve to 1, not 2.
        _action(40, 1, "PT04M00.00S", "Substitution", person_id=3, team_id=1,
                player_name="Kuzma", description="SUB: G. Antetokounmpo FOR Kuzma"),
    ]}}
    with caplog.at_level("WARNING"):
        df = _parse_pbp_payload(payload, "g1")

    giannis_back_in = df.filter(pl.col("event_num") == 40).to_dicts()[0]
    assert giannis_back_in["player2_id"] == 1  # Giannis, not Thanasis (id 2)
    assert "doesn't match playerName" not in caplog.text


def test_substitution_flags_genuine_name_mismatch(caplog):
    payload = {"game": {"gameId": "g1", "actions": [
        _action(20, 1, "PT06M15.00S", "Substitution", person_id=100, team_id=1,
                player_name="Yang", description="SUB: Thybulle FOR Hansen"),
    ]}}
    with caplog.at_level("WARNING"):
        _parse_pbp_payload(payload, "g1")
    assert "doesn't match playerName" in caplog.text


def test_substitution_learns_alias_from_own_out_side_description():
    """Real bug found on a real game: one player's structured `playerName`
    ('Yang', his family name in Chinese name order) never matches the name
    every free-text description uses for him ('Hansen', his given name).
    When he is the OUTGOING side of a sub, the description pairs 'Hansen'
    with his own already-known personId -- not a guess, since it's the same
    row's own actor. That alias must then resolve HIM as an INCOMING player
    elsewhere in the game, where 'Hansen' is the only name ever given."""
    payload = {"game": {"gameId": "g1", "actions": [
        # Hansen (id 500) is subbed OUT here -- description names him
        # "Hansen", but his own structured playerName is "Yang".
        _action(50, 1, "PT08M00.00S", "Substitution", person_id=500, team_id=1,
                player_name="Yang", description="SUB: Thybulle FOR Hansen"),
        # Later, Hansen is subbed back IN -- only "Hansen" ever appears in
        # description text, never "Yang".
        _action(90, 1, "PT03M00.00S", "Substitution", person_id=600, team_id=1,
                player_name="Clingan", description="SUB: Hansen FOR Clingan"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    second_sub = df.filter(pl.col("event_num") == 90).to_dicts()[0]
    assert second_sub["player1_id"] == 600   # OUT: Clingan
    assert second_sub["player2_id"] == 500   # IN: Hansen, resolved via alias


def test_substitution_resolves_qualified_incoming_name_never_seen_as_alias():
    """Real bug found by auditing a 400-game backfill: AJ Johnson's
    structured playerName is unqualified ('Johnson') everywhere he acts,
    but nba.com's SUB description qualifies him ('A. Johnson') to
    disambiguate from another Johnson on the roster. Unlike the
    Antetokounmpo case, he is NEVER the outgoing side of a sub in this
    game, so the alias-learning loop (which only fires on a player's own
    outgoing sub text) never registers 'a johnson'. The resolver must
    still find him via a word-subset match against the unqualified name
    learned from his own rebound action -- an exact-string lookup does
    not, and this exact gap made him vanish, then get wrongly reinstated
    as a phantom period-opener (reconstructed exactly 720s, one full
    period, regardless of his real box minutes) in 9 different real
    games."""
    payload = {"game": {"gameId": "g1", "actions": [
        # AJ Johnson (id 1642358) is only ever named "Johnson" in his own
        # actions -- never on the outgoing side of any sub.
        _action(20, 4, "PT01M14.00S", "Rebound", person_id=1642358, team_id=1,
                player_name="Johnson", description="A. Johnson REBOUND (Off:0 Def:1)"),
        _action(10, 4, "PT06M00.00S", "Substitution", person_id=999, team_id=1,
                player_name="McCollum", description="SUB: A. Johnson FOR McCollum"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    sub_row = df.filter(pl.col("event_num") == 10).to_dicts()[0]
    assert sub_row["player1_id"] == 999      # OUT: McCollum
    assert sub_row["player2_id"] == 1642358  # IN: AJ Johnson, resolved despite qualifier


def test_substitution_scoped_by_team_same_name_different_team():
    """Name resolution must be scoped to the substitution's own team_id --
    a same-named player on the OTHER team must not be picked up."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(5, 1, "PT10M00.00S", "Made Shot", person_id=777, team_id=2,
                player_name="Williams"),  # different team
        _action(6, 1, "PT09M00.00S", "Made Shot", person_id=888, team_id=1,
                player_name="Williams"),  # same team as the sub below
        _action(20, 1, "PT06M15.00S", "Substitution", person_id=100, team_id=1,
                player_name="Hartenstein", description="SUB: Williams FOR Hartenstein"),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    sub_row = df.filter(pl.col("event_num") == 20).to_dicts()[0]
    assert sub_row["player2_id"] == 888


def test_scores_are_nullable_not_zero_when_absent():
    """nba.com leaves scoreHome/scoreAway as '' on most non-scoring
    actions -- that means unknown-at-this-action, not 0-0."""
    payload = {"game": {"gameId": "g1", "actions": [
        _action(1, 1, "PT11M00.00S", "Jump Ball", score_home="", score_away=""),
    ]}}
    df = _parse_pbp_payload(payload, "g1")
    row = df.to_dicts()[0]
    assert row["home_score"] is None
    assert row["away_score"] is None
