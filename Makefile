.PHONY: install test lint db teams players games pbp box xwalk status clean

install:
	pip install -e ".[dev,scrape]"

test:
	pytest

lint:
	ruff check src tests

db:
	python -c "from nbare.warehouse.db import connect; connect(); print('schema applied')"

teams:
	nbare ingest-teams

players:
	nbare ingest-players

# Backfill one season of games. Run per season; it is resumable.
games:
	nbare ingest-games --season $(SEASON)

# The long one. ~1300 games/season, ~0.75s each => ~20 min/season.
pbp:
	nbare ingest-pbp --season $(SEASON)

# One request per game, same cost profile as pbp. Fills stg.box_player,
# which check-minutes and fit-rapm need and nothing wrote to before this.
box:
	nbare ingest-box --season $(SEASON)

xwalk:
	nbare build-xwalk

status:
	nbare status

clean:
	rm -f data/nbare.duckdb
	@echo "cache preserved at data/cache -- delete manually if you really mean it"
