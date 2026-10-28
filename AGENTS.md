# AGENTS.md

Guidance for AI coding agents (and humans) working in this repo.

## Project

RAW (`raw-agents` on PyPI) is an async Python library that gives one interface over LLM providers (Gemini, OpenAI, Anthropic, Groq, vLLM) and decision models (Jev and others via OpenRouter), plus an `Agent` with tool calling, a skill `Router`, file storage backends and a metrics protocol. Python >= 3.10 (dev uses 3.14), managed with `uv`, built with hatchling.

## Layout

```
src/RAW/
  __init__.py      # Public exports; agentic ones (Agent, LLMs) load lazily via __getattr__
  agent.py         # Agent loop: tool calls, skills, compaction, metrics, usage    [agentic]
  subagent.py      # agent_tool(): an Agent as a tool, ask_clarification/resume    [agentic]
  cli.py           # `raw-agents` command: install-skill / uninstall-skill / skill-path [base]
  bundled_skills/  # Skills shipped in the wheel (raw-agents/SKILL.md + references/)
  llms/            # BaseLLM + one module per provider                                [agentic]
  deciders/        # BaseDecider + OpenRouterDecider (Decisions API: Jev etc.)         [base]
  models/          # Pydantic/dataclass models: Message, Tool, Skill, File, Image,
                   #   decision questions/answers (decision.py), ...                   [base]
  utils/           # Logger, RequestsClient, Router (LLM or decider), count_tokens     [base]
    metrics.py     # Metrics protocol, NullMetrics, InMemoryMetrics, OpenTelemetryMetrics
    compaction.py  # History truncation/summarization used by Agent                [base]
    storage/       # Storage ABC, LocalStorage, S3Storage ([s3] extra for boto3)
tests/RAW/         # Mirrors src/RAW (note: model tests live in tests/RAW/modals/)
examples/          # Runnable example scripts (need live servers); skill folders are smoke-tested
main.py            # Interactive manual runner against a real vLLM server; not part of the package
```

## Install tiers — the most important rule

The package is split by **dependencies**, not by code:

| Install | Dependencies | What works |
|---|---|---|
| `raw-agents` | pydantic, jsonschema, httpx | `RAW.models`, `RAW.utils` (incl. storage, metrics), `RAW.deciders` |
| `raw-agents[s3]` | + boto3 | `S3Storage` |
| `raw-agents[agentic]` | + numpy, opencv-python, tiktoken, boto3 | `RAW.llms`, `RAW.Agent`, `Image`, exact token counts |
| `raw-agents[otel]` | + opentelemetry-api | `OpenTelemetryMetrics` |

Rules that follow from this:

- **Never add a top-level import of numpy, cv2, boto3/botocore or tiktoken in `RAW/__init__.py`, `RAW/models/` or `RAW/utils/`.** Import them inside the function that needs them, or under `if TYPE_CHECKING:` for annotations. When the dependency is missing, raise `ImportError` naming the extra, e.g. `pip install 'raw-agents[agentic]'` (see `models/image.py` and `utils/storage/s3.py`).
- `utils/` and `deciders/` must not import `RAW.llms` at runtime: importing `RAW.llms.base` also runs `RAW/llms/__init__.py`, which needs the agentic extra. Use `TYPE_CHECKING` (see `utils/router.py`).
- New base dependencies go in `[project] dependencies` only when they are small and needed by models/utils. Anything else goes in an extra.
- `tests/RAW/test_packaging.py` imports the base modules in a subprocess with the optional deps blocked. If it fails, you broke the boundary — fix the import, don't weaken the test.

## Commands

```bash
uv sync                          # dev env: includes the [agentic] extra via the dev group
uv run pytest -q                 # full suite, ~2s, fully offline
uv run pytest tests/RAW/utils -q # one area
uv run ruff check .              # lint (must pass; `--fix` for the safe ones)
uv run mypy src                  # types: has known pre-existing errors, don't add new ones
uv build                         # wheel + sdist
```

## Git hook

`.githooks/pre-commit` runs `ruff check` and the full test suite. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

Don't bypass it with `--no-verify` to get a commit through; fix the failure.

## Testing conventions

- **Tests never touch the network.** Mock at the `RequestsClient` layer: `patch.object(client.client, "post", new_callable=AsyncMock)` for request calls, or `patch("RAW.utils.requests.RequestsClient.post", ...)` around construction. `VLLM.__init__` sends capability probe requests, so it must be constructed inside such a patch.
- S3 tests use a `MagicMock` boto3 client and real `botocore.exceptions.ClientError` instances; local storage tests use `tmp_path`.
- Token tests substitute a fake `tiktoken` module so counts are exact and offline.
- `asyncio_mode = "auto"`: write `async def test_...` directly; `@pytest.mark.asyncio` is optional.
- The shared provider tests in `tests/RAW/llms/test_llms.py` are parametrized over providers — when adding a provider, add it to the fixture there.
- When a behaviour change makes a test fail, decide which side is wrong. Update the test only if the new behaviour is intended, and say so.

## LLM provider conventions

- Every provider subclasses `BaseLLM` and keeps its method signatures (`tests/RAW/llms/test_signatures.py` enforces names, order and defaults).
- `generate`, `chat`, `embed`, `count_tokens`, `stop` are `async`; **`info()` is sync** and must not make network calls.
- Options are a Pydantic `<Provider>Options` model; `self.capabilities` is a `List[LLMCapability]` set in `__init__`. Only send tools/images when the matching capability is present.
- Responses are normalised to `Message` (with `ToolCall`s and `Usage`); streaming yields `Message` chunks with the final usage at the end.
- HTTP goes through `RAW.utils.RequestsClient` (`is_async=True` for async calls), not raw `httpx`.

## Decider conventions

- Deciders subclass `BaseDecider` and implement `async decide(state, questions) -> Decision` and `async stop()`. They return typed answers with probabilities and **never text**; don't add text outputs to them.
- Question and answer models in `models/decision.py` mirror the OpenRouter Decisions API schema field for field (`noul`, `choice`, `score`, `box`). Check the API reference before changing them: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request.md
- Serialise questions by dropping only top-level `None` fields; `None` inside `criteria` (a Choice option with no description) is meaningful.
- Test fixtures use the real response captured in OpenRouter's Jev tutorial rather than invented shapes.
- Probabilities differ between decision models, so thresholds (e.g. `Router(decider_threshold=...)`) are per model, not universal constants.

## Style

- Match the surrounding file: type hints on public APIs, `Optional[...]`/`List[...]` in older modules, `X | None` in storage/metrics modules that use `from __future__ import annotations`.
- Catch `Exception`, never bare `except:`.
- Log through `RAW.utils.Logger`, never `print` (except `main.py`, and the user-facing output of `cli.py`).
- Keep public exports in each package's `__init__.py` / `__all__` in sync with what you add.
- Update `README.md` when public API, install instructions or extras change.
- Update the bundled skill (`src/RAW/bundled_skills/raw-agents/`) when the public API changes. `tests/RAW/test_bundled_skill.py` fails if its examples use parameters, imports or defaults that no longer exist, and `Agent`'s constructor parameters must all appear in `references/agent.md`.
