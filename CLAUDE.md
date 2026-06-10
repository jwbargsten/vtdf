# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vtdf` (very-tiny-dag-framework) is a Python library for composing computation stages into a DAG.
It is in an early, pre-implementation state: the public API in `src/vtdf/__init__.py` is mostly
stubs, and `tests/test_dag.py` specifies the intended behavior. Treat the test as the spec.

Intended design (from the test):
- A `Stage(name, fn, ctx)` wraps a function `fn(ctx)`. `ctx` is an immutable object (a frozen
  dataclass or dict) shared across stages.
- Stages compose with `>>` (via `__rshift__`/`__rrshift__`). A `list[Stage]` on either side means
  those stages run in parallel; `a >> b` means `b` depends on `a`. Composition builds a DAG and
  returns a runnable handle exposing `.run()`.

Python >=3.11, package layout is `src/`-based (`pythonpath = ["src", "tests"]`).

It is inspired by Airflow task syntax

## Commands

- `make test` — run the test suite (`uv run pytest -vvs tests/`).
- `uv run pytest tests/test_dag.py::test_dag` — run a single test.
- `make index` (alias `make idx`) — rebuild the zoekt search index (required before `zcs` works,
  and after significant file changes). `make cc` reindexes then launches `claude`.

## Code and File Search Tool: `zcs`

**IMPORTANT: This project has a code and file search index. ALWAYS use the
zcs script BEFORE using Grep/Glob/Read/Find to explore
the codebase.** The index is faster, cheaper (fewer tokens).

Run with `zcs <args>`, e.g. zcs 'lang:scala def.*unwrap'

Options:
  -n N    truncate output to first N lines (equivalent to `| head -n N`).

Stderr is merged into stdout by default — no need to add `2>&1`.

Examples:
  zcs -n 20 'lang:scala model.RoomState or RoomOverview'
  zcs 'file:Makefile'

Query Syntax (zoekt):
  file:\.py$    - filter by file pattern (regex)
  content:foo   - match only in file contents (not filenames)
  sym:Foo       - match symbol definitions (classes, defs, vals, etc.)
  type:foo      - limits result types. Possible options are type:filematch, type:file, type:filename
  lang:python   - filter by language
  -file:test    - negate (exclude matches)
  case:yes      - case-sensitive search
  "exact match" - literal phrase
  foo bar       - whitespace = implicit AND (no AND keyword exists)
  foo or bar    - boolean OR (lowercase only)
  (foo or bar)  - group expressions

To find a file use e.g.

```
zcs 'file:Makefile'
zcs 'lang:scala file:chatroutes'
```

Take into account that you might have to escape special chars for the shell.

If a query starts with `-` (e.g. negation like `-file:test`), stop arg parsing first with `--`:

```
zcs -- -file:test lang:scala def
zcs -n 20 -- -file:test foo
```

Fall back to Grep/Glob/Read/Find **only** when the zcs index doesn't cover what you need.

Skip zcs when:
- The file path is already known — use Read directly.
- Searching for uncommitted/just-edited code — the index may be stale; use Grep.

## Style Rules (apply to all output: code, comments, docs, commit messages)

- Be succinct. Say it once, say it short.
- No redundant comments. If the code is clear, don't comment it.
- No filler text, no restating the obvious, no "this function does X" before a function named X.
- When asked to "eliminate repetition" or "remove redundant comments", take it literally.
- No fluff, not fuzzy
- Don't remove FIXME/TODO comments unless the user explicitly asks. They track planned work.

## Conventions and Consistency

- Follow existing patterns in the codebase. When in doubt, match what's already there.
- Global project structure matters. Local style within a function or module is flexible.
- If a convention exists (naming, structure, patterns), follow it. Don't introduce alternatives.

## Before writing code

- Before editing any file, read it first. Before modifying a function, code/file search for all callers. Research before you edit.
- Check if a rough design or architecture decision is needed first. Ask if unclear.
- Design around data structures. Get the data model right before implementing logic around it.
- Develop the critical path first — the hard, fundamental part stripped to essentials.
- Don't introduce abstractions preemptively. Duplication is cheaper than the wrong abstraction. Let patterns emerge.
- Think about module and package structure before creating new packages.
- Don't create fine-grained packages with one class each ("categoritis"). Organise by feature, not by category.
- Don't introduce DTOs if not needed. Map directly to domain models when possible.

## Writing code

- One level of abstraction per function. Don't mix high-level orchestration with low-level details.
- Functions should fit on a screen (~80–100 lines max).
- Group code by functional cohesion (things that contribute to the same task), not by class-per-responsibility.
- Keep dependencies minimal. Don't add libraries for trivial tasks.
- No tactical DDD patterns or hexagonal architecture unless explicitly requested.
- If you don't know a library, read its docs or source on GitHub. Don't guess the API.

## Finishing a task

- Tests must pass before marking work done.

## Safety

- Never commit secrets, credentials, or API keys.
- Do not read or search files under these directories unless I explicitly ask: node_modules, build, .git, dist, __pycache__
- If a task is ambiguous, ask one clarifying question rather than guessing.
