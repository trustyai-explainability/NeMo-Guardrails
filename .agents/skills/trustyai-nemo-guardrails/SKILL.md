---
name: "trustyai-nemo-guardrails"
description: "Answers questions about the TrustyAI fork of NeMo Guardrails from a running server's admin discoverability endpoints (/admin/info, /admin/info/rails, /admin/info/actions). Use for fork-specific endpoints, guardrail flows, registered actions, or NemoGuardrails CRD integration. Trigger keywords - TrustyAI fork, what's different in this fork, capability manifest, /admin/info, /admin/info/rails, /admin/info/actions, NemoGuardrails CRD."
license: "Apache-2.0"
---

# TrustyAI NeMo Guardrails

Use this skill for questions about how the **TrustyAI fork** of NeMo Guardrails
differs from upstream, or about fork-specific endpoints and their behavioral
contracts. Answer from a running server's live admin discoverability endpoints,
not from memory or static docs. Those endpoints are generated from the actual
route table and loaded ActionDispatchers and are the source of truth for what
this deployment actually serves.

For general NeMo Guardrails product-usage questions (installation,
`config.yml`, Colang, rail catalog, etc.) that aren't specific to this fork,
use `guardrails-developer-guide` instead.

## Scope

This skill covers **Claude Code only**. It is not published to any platform
skill registry — it ships as a plugin checked into this repository and is
picked up automatically by any skill-discovering client that has the repo
checked out, per `AGENTS.md`'s "Agent Skills" section. No per-developer
install step is required beyond having the repo cloned.

## Admin Discoverability Endpoints

Paths are defined in `nemoguardrails/server/manifest.py`. They live under
`/admin/*` so the operator admin kube-rbac-proxy (Service port 444) gates them.

| Path | Returns |
| --- | --- |
| `GET /admin/info` | Fork capability document: identity, TrustyAI-tagged endpoints, documentation pointers, CRD integration metadata, built-in library rails, and server-wide actions (union across loaded configs). Does **not** include per-config payloads. |
| `GET /admin/info/rails` | Built-in library rail modules and Colang flow names for this build (same as `rails` on the parent). |
| `GET /admin/info/actions` | Actions registered on this server (same as `actions` on the parent). |

For **which config ids are mounted**, use upstream `GET /v1/rails/configs`
on the main `/v1` surface (Service port 443 when auth is enabled). That list
is not part of `/admin/info`.

## Access Model (intended vs today)

**Intended (external / workstation agents):** the TrustyAI operator exposes an
**admin Route** (or equivalent) to Service port 444. Agents call that URL for
`/admin/info*` with a bearer token that can `create`
`nemoguardrails.trustyai.opendatahub.io/<cr-name>`. The main app Route stays
on the public listener only and must not be used for `/admin/*`.

**Today:** many deployments still have **no admin Route**. The public Route
fronts only the main API. The operand `NetworkPolicy` admits ingress on the
public listener only and **never** on the admin listener (8444 / Service port
444). Cluster-internal `*.svc:444` URLs therefore do not help a workstation
agent, and `oc port-forward` to `:444` is unreliable.

Until an admin Route exists, prefer a **local `nemoguardrails server`** for
live JSON, or fall back to `CLAUDE.md` — do not treat in-cluster admin URLs as
reachable from outside the cluster.

## Finding The Endpoints

1. **Admin Route URL (preferred when present).** Read the CR annotation and
   use it only if it is a reachable external/admin hostname (not a bare
   `*.svc` name from a workstation):

   ```text
   oc get nemoguardrails <cr-name> -n <namespace> \
     -o jsonpath='{.metadata.annotations.trustyai\.opendatahub\.io/nemo-guardrails-manifest-url}'
   ```

   If the value is still `https://…svc:444/…` (or similar cluster-DNS only),
   the operator has not exposed admin yet — skip to (2). When auth is enabled,
   send `Authorization: Bearer <token>` with `create` on that CR. Never disable
   TLS verification.

2. **Local `nemoguardrails server`.** Fetch
   `http://<host>:<port>/admin/info` (and `/admin/info/rails`,
   `/admin/info/actions`). No proxy or token. Use this for workstation agents
   when no admin Route is available.

3. **Do not rely on port-forward to Service port 444** as the primary path.
   It may work on some clusters and hang on others once NetworkPolicy is
   present. If the user already has a working forward, you may use it; if it
   fails once, stop and use (2) or the fallback below.

## Handling Failures

A failed fetch is not a dead end. Tell the user which step failed, quote the
error, and use the guidance below.

- **No admin Route / only `*.svc:444` from a workstation.** Expected until the
  operator exposes admin. Use a local server or `CLAUDE.md`.
- **Connection refused, DNS failure, timeout, or port-forward hang against
  `:444`.** Assume admin is not ingress-reachable (no Route / NetworkPolicy),
  not that the pod is necessarily down. Prefer local server or `CLAUDE.md`.
- **404 on an admin URL.** Image may predate these endpoints, or the request
  hit the main port (443), which returns 404 outside `/v1/*`. Don't second-guess
  with the checkout's path constants; the deployment may be on a different
  version.
- **401.** Token missing, malformed, or expired — ask the user to `oc login`
  and retry.
- **403.** Token valid but not authorized — ask
  `oc auth can-i create nemoguardrails.trustyai.opendatahub.io/<cr-name> -n <namespace>`.
  View-only users cannot read admin endpoints. Retrying the same token will not
  help.
- **TLS error.** Never disable certificate verification. Report the error
  verbatim.
- **Annotation missing.** Older operator — check
  `security.opendatahub.io/enable-auth` and whether an admin Route exists at
  all.
- **Admin still unreachable.** Fall back to the "Key fork changes" section of
  `CLAUDE.md`, and say this is best-effort and may be out of date. Never invent
  fork-specific behavior. Do not invent
  `spec.networkPolicy.additionalIngress` peers to open the admin listener
  unless a maintainer has documented that as supported.
- **Actions list is empty.** Configs may have failed to load, or
  `rails_config_path` may be missing. Search logs for
  `Skipping actions for config` or `Rails config path ... is missing`. Rails
  and endpoint sections of `/admin/info` may still be usable.

## Using The Content

1. Start with `GET /admin/info` for fork identity, endpoints, docs, rails, and
   actions. Use `/admin/info/rails` or `/admin/info/actions` when only that
   slice is needed.
2. For mounted config ids, call `GET /v1/rails/configs` on the main `/v1`
   surface, not `/admin/info`.
3. To learn what response fields mean, read
   `nemoguardrails/server/schemas/manifest.py` in the checkout
   (`CapabilityManifest`, `ManifestAction`, etc.) and tell the user the
   checkout may not match the deployment. Do **not** expect `/openapi.json` on
   the admin port: kube-rbac-proxy-admin only allows `/admin/*`. On a local
   unproxied server, `/openapi.json` on that process is fine if needed.
4. Quote the response field values directly (exact endpoint paths and
   behavioral contract text) instead of paraphrasing from memory or static
   docs.

## Related Skills

- Use `guardrails-developer-guide` for upstream/general product-usage
  questions.
- Use `guardrails-developer-create-guardrails` when the user wants to build a
  guardrails configuration rather than ask about this fork's own behavior.
