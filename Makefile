.DEFAULT_GOAL := all


NAME=$(shell basename `pwd`)
DOCS_DIR="docs"


help:  ## Display help and quit
	@echo Makefile for the $(NAME) package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

all: install test  ## Install the package and run tests

clean_build:
	rm -rf build/

clean_dist:
	rm -rf dist/

clean_eggs:
	rm -rf *.egg-info

clean_mypy:
	rm -rf .mypy_cache/

clean_docs:
	rm -rf docs/build/*

clean_tests:
	rm -rf /tmp/pytest*

clean: clean_dist clean_eggs clean_build clean_mypy clean_docs  ## Remove build, mypy cache, tests and docs

_install:
	# python setup.py sdist
	# python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	# python -m pip install dist/*-py3-none-any.whl --user --upgrade
	python -m pip install .

install:  ## Install the package
	${MAKE} clean
	${MAKE} _install
	${MAKE} clean

docs:  ## Build the documentation
	${MAKE} -C $(DOCS_DIR) html
	xdg-open $(DOCS_DIR)/build/html/index.html

test:  ## Run the tests
	python -m pytest $(NAME)/ -m "not slow"

backup_time:
	echo "Last backup: " `date` >> _backup_time
	chmod 700 _backup_time

_sync:
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/$(NAME)

sync: _sync backup_time ## [dev] Sync data/code to SCU server

build: test
		python setup.py sdist bdist_wheel

pypitest: build
		twine \
				upload \
				-r pypitest dist/*

pypi: build
		twine \
				upload \
				dist/*

.PHONY : clean_build clean_dist clean_eggs clean_mypy clean_docs clean_tests \
clean _install install clean_docs docs test backup_time _sync sync \
build pypitest pypi
