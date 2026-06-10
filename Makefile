PATH:=$(PATH):$(abspath script)
export PATH

index:
	rm -rf ./search
	zoekt-index -require_ctags -index search -ignore_dirs generated,search,output,.git,node_modules,.idea,target,dist,build,.claude,.vscode src/ tests/

idx: index

cc: index
	claude

test:
	uv run pytest -vvs tests/

lint: ## lint the source code
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fmt: ## format the source code with ruff
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/
