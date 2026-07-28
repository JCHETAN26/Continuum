Continuum uses the shared Alembic migration environment in `packages/shared/alembic`.

Run service migrations from the repository root:

```bash
uv run alembic -c packages/shared/alembic.ini upgrade head
```
