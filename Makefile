.PHONY: build commit quality style test check-hf-modeling check-agent-docs

check_dirs := tasks tests veomni docs scripts/sync_hf_modeling.py

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

check-hf-modeling:
	python3 scripts/sync_hf_modeling.py check

check-agent-docs:
	python3 scripts/ci/docs/check_agent_doc_paths.py
