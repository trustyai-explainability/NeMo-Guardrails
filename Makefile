PLATFORMS ?= linux/$(TARGETARCH)
TARGETARCH ?= amd64

DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
IMAGE_BUILD_EXTRA_OPTS ?=

IMAGE_REGISTRY ?= quay.io/trustyai
IMAGE_NAME := nemo-guardrails-server
IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)

GIT_COMMIT_SHA ?= $(shell git rev-parse HEAD)
GIT_TAG ?= $(or $(shell git describe --abbrev=0 2>/dev/null),$(shell git rev-parse --short HEAD))
BUILD_REF ?= $(or $(shell git describe --abbrev=0 2>/dev/null),$(GIT_TAG)) # release tag or fallback to git commit hash

# The name of the kind cluster to use for the "kind-load" target.
KIND_CLUSTER ?= kind

.PHONY: all test tests test_watch test_coverage test_profile docs docs-strict docs-serve docs-update-cards docs-check-cards docs-watch-cards pre_commit help

# Default target executed when no specific target is provided to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/

test:
	poetry run pytest $(TEST_FILE)

tests:
	poetry run pytest $(TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

test_coverage:
	poetry run pytest --cov=$(TEST_FILE) --cov-report=term-missing

test_profile:
	poetry run pytest -vv tests/ --profile-svg

docs:
	poetry run sphinx-build -b html docs _build/docs

docs-strict:
	poetry run sphinx-build -b html -W --keep-going docs _build/docs

docs-serve:
	cd docs && poetry run sphinx-autobuild . _build/html --port 8000 --open-browser

docs-update-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py

docs-check-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py --dry-run

docs-watch-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py watch

docs-check-redirects:
	cd docs && poetry run python scripts/validate_redirects.py

pre_commit:
	pre-commit install
	pre-commit run --all-files

# BUILD
.PHONY: image-build
image-build: # Build the image using Docker buildx.
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--file Dockerfile.server \
		--platform=$(PLATFORMS) \
		--build-arg COMMIT_SHA=${GIT_COMMIT_SHA} \
		--build-arg BUILD_REF=${BUILD_REF} \
		$(PUSH) \
		$(LOAD) \
		$(IMAGE_BUILD_EXTRA_OPTS) ./

# Build the container image for the server
.PHONY: image-local-build
image-local-build: # Build the image using Docker buildx
	set -e; \
	builder=$$($(DOCKER_BUILDX_CMD) create --use); \
	trap '$(DOCKER_BUILDX_CMD) rm -f "$$builder"' EXIT; \
	$(MAKE) image-build PUSH="$(PUSH)" LOAD="$(LOAD)"

.PHONY: image-local-push
image-local-push: # Push the image to the local Docker registry
image-local-push: PUSH=--push # Build the image for local development and push it to $IMAGE_REPO.
image-local-push: image-local-build

.PHONY: image-kind
image-kind: LOAD=--load
image-kind: image-build # Build the image and load it to kind cluster $KIND_CLUSTER ("kind" by default)
	kind load docker-image $(IMAGE_TAG) --name $(KIND_CLUSTER)

# HELP

help:
	@echo '----'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in given file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_coverage                - run unit tests with coverage'
	@echo 'docs                         - build docs, if you installed the docs dependencies'
	@echo 'docs-strict                  - build docs with warnings as errors (used in CI)'
	@echo 'docs-serve                   - serve docs locally with auto-rebuild on changes'
	@echo 'docs-update-cards            - update grid cards in index files from linked pages'
	@echo 'docs-check-cards             - check if grid cards are up to date (dry run)'
	@echo 'docs-watch-cards             - watch for file changes and auto-update cards'
	@echo 'docs-check-redirects         - validate that all redirect targets exist'
	@echo 'pre_commit                   - run pre-commit hooks'
	@echo 'image-build                  - build the image using Docker buildx'
	@echo 'image-local-build            - build the image using Docker buildx for local development'
	@echo 'image-local-push             - build the image and push it to the local Docker registry'
	@echo 'image-kind                   - build the image and load it to kind cluster $KIND_CLUSTER ("kind" by default)'
