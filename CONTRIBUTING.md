# Contributing to FlowCast

Thank you for your interest in contributing to FlowCast! This guide covers everything you need to know to contribute cleanly.

---

## Getting Started

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/FlowCast.git
   cd FlowCast
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. **Download the dataset and verify everything works:**
   ```bash
   .venv/bin/python run.py --check-only
   .venv/bin/python -m pytest tests/ -v
   ```

---

## Development Workflow

### Branching

- Always branch from `main`.
- Use descriptive branch names: `feat/improved-gru`, `fix/dataloader-ssl`, `docs/literature-section`.

### Code Style

- All Python code must comply with **PEP 8**.
- Type annotations are required for all public functions.
- Module-level and function-level docstrings are mandatory for all new source files.
- Run `python -m py_compile src/**/*.py` before committing to catch syntax errors.

### Testing

All changes must maintain or increase test coverage. Before opening a pull request:

```bash
# Run the full test suite (must all pass)
.venv/bin/python -m pytest tests/ -v
```

New features require accompanying tests in `tests/`. Bug fixes should include a regression test that would have caught the bug.

### Commit Messages

Use **Conventional Commits** format:

| Prefix | Use for |
| --- | --- |
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation-only changes |
| `test:` | Adding or updating tests |
| `refactor:` | Code restructuring without behaviour change |
| `chore:` | Maintenance (dependencies, CI, gitignore) |

Examples:
```
feat: add per-sensor R² breakdown to evaluation metrics
fix: handle edge case when val_size equals train length
docs: expand GRU architecture section in methodology.md
test: add batch consistency tests for FlowCastPredictor
```

---

## Project Structure

```
FlowCast/
├── src/
│   ├── data/          # Dataset download, loading, and preprocessing
│   ├── models/        # FlowCastGRU, RandomForest baseline, inference
│   └── evaluation/    # Regression & classification metrics, plotting
├── tests/             # pytest test suite (must all pass before PR)
├── docs/              # Academic documentation
├── artifacts/         # Generated evaluation figures, metrics JSON
│   ├── figures/
│   └── metrics/
├── data/raw/          # Raw dataset files (auto-downloaded, gitignored)
├── app.py             # Streamlit interactive dashboard
├── run.py             # Single-command launcher
└── requirements.txt   # Pinned dependency list
```

---

## Academic Integrity Note

FlowCast is an academic prototype. If you extend it:
- Do **not** claim the system controls real traffic signals.
- Do **not** fabricate benchmark results.
- Clearly attribute any external model or algorithm you include.
- Keep the UCI dataset citation in any derived work.

---

## Opening a Pull Request

1. Ensure `pytest tests/ -v` passes with no failures.
2. Ensure your branch is up to date with `origin/main`.
3. Fill in the PR template describing: *what changed*, *why it changed*, and *how it was tested*.
4. Tag a maintainer for review.

---

## Reporting Issues

Please include:
- Python version (`python --version`)
- Operating system
- Exact error message and traceback
- Steps to reproduce

Open an issue at <https://github.com/mdzaidibrahim/FlowCast/issues>.
