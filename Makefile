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

.PHONY: help
.PHONY: test test-parallel test-serial test-benchmark test-watch test-coverage test-profile warm-fastembed-cache
.PHONY: docs-fern docs-fern-strict docs-fern-live docs-fern-preview-watch docs-fern-generate-sdk docs-fern-fix-empty-links docs-check-links docs-check-redirects
.PHONY: pre-commit
.PHONY: image-build image-local-build image-local-push image-kind

.DEFAULT_GOAL := help

TEST ?=
ARGS ?=
WORKERS ?= auto
# pytest-xdist --dist strategy for $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST).
# worksteal dynamically rebalances queued tests; override DIST when debugging or grouping matters.
DIST ?= worksteal

PYTEST ?= poetry run pytest
# These targets assume a Unix-like shell for env -u; use bash, Git Bash, or WSL on Windows.
UNIT_TEST_ENV ?= env -u OPENAI_API_KEY -u NVIDIA_API_KEY \
	-u LIVE_TEST -u LIVE_TEST_MODE -u TEST_LIVE_MODE

FASTEMBED_CACHE ?= .cache/fastembed
FASTEMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
FASTEMBED_ENV ?= env FASTEMBED_CACHE_PATH=$(FASTEMBED_CACHE)
FERN_STAGING_INSTANCE ?= nvidia-nemo-guardrails-staging.docs.buildwithfern.com/nemo/guardrails

test:
	$(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST)

test-parallel: test

test-serial:
	$(PYTEST) $(ARGS) $(TEST)

test-benchmark:
	$(PYTEST) $(ARGS) benchmark/tests

test-watch:
	poetry run ptw --snapshot-update --now . -- -vv $(ARGS) $(TEST)

test-coverage:
	$(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) --cov=nemoguardrails --cov-report=xml:coverage.xml $(ARGS) $(TEST)

test-profile:
	$(PYTEST) -vv --profile-svg $(ARGS) $(TEST)

warm-fastembed-cache:
	$(FASTEMBED_ENV) poetry run python -c 'from fastembed import TextEmbedding; model = TextEmbedding("$(FASTEMBED_MODEL)"); next(model.embed(["warmup"]))'

docs-fern: docs-fern-strict

docs-fern-strict: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" check

docs-fern-live: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs dev

docs-fern-publish-staging: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" generate --docs --instance "$(FERN_STAGING_INSTANCE)"

docs-fern-preview-watch: docs-fern-generate-sdk
	node scripts/watch-fern-preview.mjs

docs-fern-generate-sdk:
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs md generate --library guardrails-python-sdk
	node scripts/normalize-fern-sdk-reference.mjs

docs-fern-fix-empty-links:
	node scripts/fix-empty-fern-links.mjs

docs-check-links:
	bash scripts/check-docs-links.sh --local-only

docs-check-redirects:
	cd docs && poetry run python scripts/validate_redirects.py

pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

# BUILD
image-build:
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--file Dockerfile.server \
		--platform=$(PLATFORMS) \
		--build-arg COMMIT_SHA=${GIT_COMMIT_SHA} \
		--build-arg BUILD_REF=${BUILD_REF} \
		$(PUSH) \
		$(LOAD) \
		$(IMAGE_BUILD_EXTRA_OPTS) ./

image-local-build:
	set -e; \
	builder=$$($(DOCKER_BUILDX_CMD) create --use); \
	trap '$(DOCKER_BUILDX_CMD) rm -f "$$builder"' EXIT; \
	$(MAKE) image-build PUSH="$(PUSH)" LOAD="$(LOAD)"

image-local-push: PUSH=--push
image-local-push: image-local-build

image-kind: LOAD=--load
image-kind: image-build
	kind load docker-image $(IMAGE_TAG) --name $(KIND_CLUSTER)

help:
	@printf '%s\n' \
		'' \
		'Usage:' \
		'  make test [TEST=path] [WORKERS=auto] [ARGS="-q --tb=short"]' \
		'  make test-serial [TEST=path] [ARGS="-q"]' \
		'  make test-benchmark [ARGS="-q"]' \
		'  make test-parallel [TEST=path] [WORKERS=auto] [ARGS="-q --tb=short"]' \
		'  make test-watch [TEST=path]' \
		'' \
		'Tests:' \
		'  test                  Run pytest.ini testpaths with pytest-xdist' \
		'  test-parallel         Alias for test' \
		'  test-serial           Run pytest without xdist or env filtering' \
		'  test-benchmark        Run benchmark tooling tests' \
		'  test-watch            Run pytest in watch mode' \
		'  test-coverage         Run pytest with coverage' \
		'  test-profile          Run pytest with profiling' \
		'  warm-fastembed-cache  Prime the repo-local FastEmbed cache' \
		'' \
		'Docs:' \
		'  docs-fern             Check Fern docs using the pinned Fern CLI' \
		'  docs-fern-strict      Check Fern docs using the pinned Fern CLI' \
		'  docs-fern-live        Serve Fern docs locally' \
		'  docs-fern-publish-staging Publish Fern docs to the staging instance' \
		'  docs-fern-preview-watch Watch and publish Fern preview for the current branch' \
		'  docs-fern-generate-sdk Regenerate Python SDK reference pages with Fern' \
		'  docs-fern-fix-empty-links Replace empty Markdown links with titles from Fern navigation' \
		'  docs-check-links     Validate Markdown and MDX links locally' \
		'  docs-check-redirects  Validate docs redirects' \
		'' \
		'Build:' \
		'  image-build           Build the container image using Docker buildx' \
		'  image-local-build     Build the image for local development' \
		'  image-local-push      Build and push to the local Docker registry' \
		'  image-kind            Build and load to kind cluster $$KIND_CLUSTER' \
		'' \
		'Maintenance:' \
		'  pre-commit            Install and run pre-commit hooks'
