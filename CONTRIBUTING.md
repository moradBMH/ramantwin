# Contributing

## Workflow
- `main` is always green. Work on short branches named `<type>/<topic>`, one pull request per branch.
- Merge with **Rebase and merge** after CI passes and the review is done.
- Interface changes (`docs/architecture.md`) go in their own pull request, before the code that needs them.

## Commits
[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative subject`, 72 characters max.

- Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`, `ci`.
- Scopes: `twin`, `pipeline`, `fleet`, `ml`, `qc`, `api`, `web`, `docs`, `repo`.
- Small, atomic commits that each pass the tests.

## Web
npm only, with `web/package-lock.json` committed (CI runs `npm ci`). Generated data in
`web/public/data/` is committed; `/data/` and `/artifacts/` at the root are not.

## Checks
```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
cd web && npm ci && npm run build
```
