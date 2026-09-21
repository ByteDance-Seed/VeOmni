.PHONY: build commit quality style test patchgen check-patchgen check-agent-docs

check_dirs := tasks tests veomni docs

build:
	python3 setup.py sdist bdist_wheel

commit:
	pre-commit install
	pre-commit run --all-files

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	pytest tests/

patchgen:
	patchgen --all --diff

check-patchgen:
	patchgen --check

check-agent-docs:
	python3 scripts/ci/check_agent_doc_paths.py
