.PHONY: build commit quality style test patchgen check-patchgen

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
	python -m veomni.patchgen.check_patchgen --fix

check-patchgen:
	python -m veomni.patchgen.check_patchgen
