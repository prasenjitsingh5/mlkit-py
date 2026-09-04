# Contributing

This is a personal project, but fixes and small improvements are welcome.

1. Open an issue first for anything larger than a bug fix.
2. Fork, create a branch, and keep each pull request to one change.
3. Run `uv run ruff check .`, `uv run ruff format .` and `uv run pytest` before pushing. CI runs the same from the committed `uv.lock`.
4. Add or update tests for any behaviour change and note it in `CHANGELOG.md`.
5. By opening a pull request you confirm the work is yours and is licensed under the MIT License like the rest of the project. See [DISCLAIMER.md](DISCLAIMER.md).
