SHELL := /usr/bin/env bash

default: help

.PHONY: help

help: # Show help for each of the Makefile recipes.
	@grep -E '^[^:]+:.*#' $(firstword $(MAKEFILE_LIST)) \
	| grep -v $$'\t' \
	| sort \
	| while IFS=: read target msg; \
	do \
		echo -e $(FMT-BOLD)$(FMT-YELLOW)$$target$(FMT-RESET): $${msg/*#/}; \
	done


###############
# Unit tests
###############

test-python: # Run python tests
	@python -m pytest --verbose --ignore=third_party python/oxvox/tests

test: # Run all tests
	@make test-python
