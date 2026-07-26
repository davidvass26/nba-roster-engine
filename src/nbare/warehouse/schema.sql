-- nbare warehouse schema
--
-- Three layers:
--   raw_*     verbatim source payloads, never edited, safe to drop+rebuild
--   stg_*     typed / normalized / id-resolved
--   mart_*    analysis-ready tables the modelling code reads
--
-- Everything is rebuildable from data/cache, so no layer is precious.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS mart;

-- =====================================================================
-- RAW
-- =====================================================================

-- Every nba.com response lands here first, keyed by the request that
-- produced it. Lets us replay parsing changes without refetching.
CREATE TABLE IF NOT EXISTS raw.nba_response (
    request_hash   VARCHAR PRIMARY KEY,
    endpoint       VARCHAR NOT NULL,
    params_json    VARCHAR NOT NULL,
    fetched_at     TIMESTAMP NOT NULL,
    payload_json   VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.contract_page (
    source         VARCHAR NOT NULL,   -- 'spotrac' | 'bbref'
    url            VARCHAR NOT NULL,
    fetched_at     TIMESTAMP NOT NULL,
    html           VARCHAR NOT NULL,
    PRIMARY KEY (source, url)
);

-- =====================================================================
-- STAGING
-- =====================================================================

CREATE TABLE IF NOT EXISTS stg.player (
    nba_player_id  BIGINT PRIMARY KEY,
    full_name      VARCHAR NOT NULL,
    first_name     VARCHAR,
    last_name      VARCHAR,
    birthdate      DATE,
    position       VARCHAR,
    height_in      SMALLINT,
    weight_lb      SMALLINT,
    draft_year     SMALLINT,
    draft_round    SMALLINT,
    draft_number   SMALLINT,
    from_year      SMALLINT,
    to_year        SMALLINT,
    is_active      BOOLEAN
);

CREATE TABLE IF NOT EXISTS stg.team (
    nba_team_id    BIGINT PRIMARY KEY,
    abbreviation   VARCHAR NOT NULL,
    nickname       VARCHAR,
    city           VARCHAR,
    full_name      VARCHAR
);

-- Crosswalk between id systems. Populated by crosswalk.build, with
-- manual overrides applied last so they always win.
CREATE TABLE IF NOT EXISTS stg.player_xwalk (
    nba_player_id  BIGINT PRIMARY KEY,
    bbref_slug     VARCHAR,
    spotrac_slug   VARCHAR,
    match_method   VARCHAR NOT NULL,   -- 'exact' | 'fuzzy' | 'manual'
    match_score    DOUBLE,
    resolved_at    TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS stg.game (
    game_id        VARCHAR PRIMARY KEY,
    season         VARCHAR NOT NULL,
    season_type    VARCHAR NOT NULL,   -- 'Regular Season' | 'Playoffs'
    game_date      DATE NOT NULL,
    home_team_id   BIGINT NOT NULL,
    away_team_id   BIGINT NOT NULL,
    home_pts       SMALLINT,
    away_pts       SMALLINT
);

-- Play-by-play, one row per event. The substitution events in here are
-- what Stage 2 reconstructs lineups from.
CREATE TABLE IF NOT EXISTS stg.pbp_event (
    game_id            VARCHAR NOT NULL,
    event_num          INTEGER NOT NULL,
    period             SMALLINT NOT NULL,
    clock_seconds_left DOUBLE,          -- seconds remaining in period
    event_type         VARCHAR,
    event_action_type  VARCHAR,
    description        VARCHAR,
    team_id            BIGINT,
    player1_id         BIGINT,
    player2_id         BIGINT,
    player3_id         BIGINT,
    home_score         SMALLINT,
    away_score         SMALLINT,
    PRIMARY KEY (game_id, event_num)
);

-- Official per-game minutes, used to VALIDATE lineup reconstruction.
-- If reconstructed minutes disagree with these by more than a few
-- seconds, the stint builder has a bug. Do not skip this check.
CREATE TABLE IF NOT EXISTS stg.box_player (
    game_id        VARCHAR NOT NULL,
    nba_player_id  BIGINT NOT NULL,
    nba_team_id    BIGINT NOT NULL,
    seconds_played INTEGER,
    pts            SMALLINT,
    reb            SMALLINT,
    ast            SMALLINT,
    started        BOOLEAN,
    PRIMARY KEY (game_id, nba_player_id)
);

-- =====================================================================
-- CONTRACTS  (Stage 1 depends on these)
-- =====================================================================

CREATE TABLE IF NOT EXISTS stg.contract (
    contract_id        VARCHAR PRIMARY KEY,
    nba_player_id      BIGINT NOT NULL,
    nba_team_id        BIGINT,
    signed_date        DATE,
    total_value        BIGINT,
    years              SMALLINT,
    has_trade_bonus    BOOLEAN DEFAULT FALSE,
    trade_bonus_amount BIGINT DEFAULT 0,
    no_trade_clause    BOOLEAN DEFAULT FALSE,
    source             VARCHAR
);

-- One row per contract-season. The option/guarantee columns are the
-- footnotes on Spotrac tables that everyone ignores and that determine
-- whether a salary is actually tradeable or a cap hold.
CREATE TABLE IF NOT EXISTS stg.contract_year (
    contract_id      VARCHAR NOT NULL,
    season           VARCHAR NOT NULL,
    nba_player_id    BIGINT NOT NULL,
    cap_hit          BIGINT NOT NULL,
    guaranteed       BIGINT NOT NULL,
    option_type      VARCHAR,          -- NULL | 'player' | 'team' | 'ETO'
    is_two_way       BOOLEAN DEFAULT FALSE,
    is_rookie_scale  BOOLEAN DEFAULT FALSE,
    incentives_likely   BIGINT DEFAULT 0,
    incentives_unlikely BIGINT DEFAULT 0,
    PRIMARY KEY (contract_id, season)
);

-- Bird rights tier: determines which exception a team can re-sign with.
CREATE TABLE IF NOT EXISTS stg.player_rights (
    nba_player_id  BIGINT NOT NULL,
    season         VARCHAR NOT NULL,
    nba_team_id    BIGINT NOT NULL,
    bird_tier      VARCHAR NOT NULL,  -- 'full' | 'early' | 'non' | 'none'
    years_of_service SMALLINT,
    PRIMARY KEY (nba_player_id, season)
);

-- Real transactions. This table IS the Stage 1 validation set: replay
-- each one through the rule engine and it must come back legal.
CREATE TABLE IF NOT EXISTS stg.transaction (
    transaction_id VARCHAR PRIMARY KEY,
    txn_date       DATE NOT NULL,
    season         VARCHAR NOT NULL,
    txn_type       VARCHAR NOT NULL,  -- 'trade'|'signing'|'waive'|'claim'
    description    VARCHAR,
    source_url     VARCHAR
);

CREATE TABLE IF NOT EXISTS stg.transaction_leg (
    transaction_id VARCHAR NOT NULL,
    leg_num        SMALLINT NOT NULL,
    nba_player_id  BIGINT,
    from_team_id   BIGINT,
    to_team_id     BIGINT,
    cash_amount    BIGINT DEFAULT 0,
    pick_detail    VARCHAR,
    PRIMARY KEY (transaction_id, leg_num)
);

-- =====================================================================
-- MARTS
-- =====================================================================

-- Contiguous possessions with a fixed 10-man lineup. Stage 2's design
-- matrix is built directly off this.
CREATE TABLE IF NOT EXISTS mart.stint (
    stint_id       VARCHAR PRIMARY KEY,
    game_id        VARCHAR NOT NULL,
    period         SMALLINT NOT NULL,
    start_seconds  DOUBLE NOT NULL,
    end_seconds    DOUBLE NOT NULL,
    off_team_id    BIGINT,
    home_lineup    BIGINT[] NOT NULL,
    away_lineup    BIGINT[] NOT NULL,
    possessions    DOUBLE,
    home_pts       SMALLINT,
    away_pts       SMALLINT
);

-- Team cap sheet snapshot, the input object for the Stage 1 engine.
CREATE TABLE IF NOT EXISTS mart.cap_sheet (
    nba_team_id    BIGINT NOT NULL,
    season         VARCHAR NOT NULL,
    as_of          DATE NOT NULL,
    cap_payroll    BIGINT NOT NULL,
    apron_payroll  BIGINT NOT NULL,
    roster_count   SMALLINT NOT NULL,
    two_way_count  SMALLINT NOT NULL,
    PRIMARY KEY (nba_team_id, season, as_of)
);

CREATE TABLE IF NOT EXISTS mart.trade_exception (
    exception_id   VARCHAR PRIMARY KEY,
    nba_team_id    BIGINT NOT NULL,
    season         VARCHAR NOT NULL,
    amount         BIGINT NOT NULL,
    created_date   DATE NOT NULL,
    expires_date   DATE NOT NULL,
    used_amount    BIGINT DEFAULT 0
);
