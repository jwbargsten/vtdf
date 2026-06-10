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
