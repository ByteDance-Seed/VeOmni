.PHONY: build commit quality style test patchgen check-patchgen

check_dirs := tasks tests veomni docs scripts

build:
	python3 -m build

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
