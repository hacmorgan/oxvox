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

test-rust: # Run rust unit tests
	@cargo test

test: # Run all tests (rust unit tests, then python tests)
	@make test-rust
	@make test-python


###############
# Linting
###############

lint: # Check rust formatting and run clippy with warnings denied
	@cargo fmt --check
	@cargo clippy --all-targets -- -D warnings
