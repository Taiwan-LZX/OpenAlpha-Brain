# Contributing to OpenAlpha-Brain

Thanks for your interest in contributing! OpenAlpha-Brain is an autonomous alpha research platform for WorldQuant BRAIN, and we welcome contributions of all kinds.

## Code of Conduct

Be respectful, constructive, and collaborative. Assume good intent when reviewing others' work.

---

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Clone and Install

```bash
git clone https://github.com/openalpha-brain/openalpha-brain.git
cd openalpha-brain
pip install -e ".[dev,web]"
```

This installs the package in **editable mode** (`-e`) so code changes take effect immediately, plus all development and web server dependencies:

| Extra | Contents |
|---|---|
| `dev` | pytest, pytest-asyncio, ruff, mypy |
| `web` | FastAPI, uvicorn |

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` to provide your LLM API key. BRAIN credentials are optional for local-only development — use `--no-brain` to skip BRAIN submission during testing.

---

## Code Style

We use **ruff** for linting and formatting, and **mypy** for static type checking. Configurations are in `pyproject.toml`.

### Lint

```bash
ruff check src/ tests/
```

### Format

```bash
ruff format src/ tests/
```

Run with `--check` to verify without modifying:

```bash
ruff format --check src/ tests/
```

### Type Check

```bash
mypy src/
```

### Pre-Commit Hooks (Recommended)

Pre-commit hooks run checks automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

---

## Running Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_validator.py -v
```

Run a subset of tests by keyword:

```bash
pytest tests/ -v -k "mutation"
```

Run the 10-cycle smoke test (fast):

```bash
pytest tests/test_e2e.py -v -k "10cycle"
```

> **Note:** Tests that interact with the BRAIN API require valid credentials in `.env` and `BRAIN_SUBMIT_ENABLED=true`. These are automatically skipped if credentials are missing.

---

## Project Structure

The codebase follows an **src-layout** with clean module boundaries:

```
src/openalpha_brain/
├── cli/            # CLI entry points (`openalpha` command)
├── config/         # Settings loaded from .env
├── core/           # Loop engine, models, pipeline orchestration
├── generation/     # LLM prompts, alpha parsing, expression generation
├── validation/     # Syntax validation, AST repair, stability checks
├── evolution/      # Mutation strategies, crossover, trajectory mutation
├── services/       # External API clients (BRAIN, LLM, HTTP pool)
├── knowledge/      # RAG engine, vector index, skill library
├── learning/       # Experience distillation, MAB, parametric optimization
├── agents/         # Multi-agent coordination
├── data/           # Static data files (operators.json, grammar, etc.)
└── utils/          # Logging, auditing, market state helpers
```

---

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.

2. **Make your changes** following the code style rules above.

3. **Add tests** for new functionality. Existing tests must continue to pass.

4. **Run the full test suite:**
   ```bash
   pytest tests/ -v
   ```

5. **Run linting and type checks:**
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   mypy src/
   ```

6. **Write a clear PR description:**
   - What problem does this solve?
   - What approach did you take?
   - Any breaking changes?
   - Link to related issues.

7. **Submit the PR** against the `main` branch.

### PR Review Criteria

- Code follows project conventions and style guides
- New code has corresponding tests
- No regressions in existing tests
- Changes are focused and well-scoped

---

## Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output or error messages

Use the [GitHub Issues](https://github.com/openalpha-brain/openalpha-brain/issues) tracker.

---

## Questions?

Open a [GitHub Discussion](https://github.com/openalpha-brain/openalpha-brain/discussions) or ask in the community channels.