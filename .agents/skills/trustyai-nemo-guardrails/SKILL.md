---
name: "trustyai-nemo-guardrails"
description: "Answers questions about the TrustyAI fork of NeMo Guardrails from a running server's /admin/info capability manifest. Use for fork-specific endpoints, guardrail flows, available configs, or NemoGuardrails CRD integration. Trigger keywords - TrustyAI fork, what's different in this fork, capability manifest, /admin/info, NemoGuardrails CRD."
license: "Apache-2.0"
---

# TrustyAI NeMo Guardrails

Use this skill for questions about how the **TrustyAI fork** of NeMo Guardrails
differs from upstream, or about fork-specific endpoints and their behavioral
contracts. Answer from a running server's live manifest, not from memory or
static docs, since the manifest is generated from the actual route table at
startup and is the source of truth for what this deployment actually serves.

For general NeMo Guardrails product-usage questions (installation,
`config.yml`, Colang, rail catalog, etc.) that aren't specific to this fork,
use `guardrails-developer-guide` instead.

## Scope

This skill covers **Claude Code only**. It is not published to any platform
skill registry — it ships as a plugin checked into this repository and is
picked up automatically by any skill-discovering client that has the repo
checked out, per `AGENTS.md`'s "Agent Skills" section. No per-developer
install step is required beyond having the repo cloned.

## Finding The Manifest

The server defines the manifest path as `MANIFEST_PATH` in
`nemoguardrails/server/manifest.py`. It is currently `/admin/info`, which is
temporary until the EvalHub team agrees on a platform-wide Agent
Discoverability contract.

1. **Local dev server**: fetch `http://<host>:<port>/admin/info` from the
   running `nemoguardrails server`. No proxy or token is involved.
2. **In-cluster deployment (TrustyAI operator)**: read the URL from the
   `NemoGuardrails` CR's `trustyai.opendatahub.io/nemo-guardrails-manifest-url`:

   ```text
   oc get nemoguardrails <cr-name> -n <namespace> \
     -o jsonpath='{.metadata.annotations.trustyai\.opendatahub\.io/nemo-guardrails-manifest-url}'
   ```

   The URL depends on whether the CR sets
   `security.opendatahub.io/enable-auth: "true"`:

   - **Auth enabled**: `https://<cr-name>.<namespace>.svc:444/admin/info`.
     Port 444 is the admin port, behind kube-rbac-proxy-admin. The caller
     needs a bearer token allowed to `create`
     `nemoguardrails.trustyai.opendatahub.io/<cr-name>` in that namespace.
     Port 443 serves only `/v1/*` and cannot reach the manifest.
   - **Auth disabled**: `http://<cr-name>.<namespace>.svc/admin/info`. No
     proxy or token is involved.

   Both are cluster-internal Service URLs. NeMo Guardrails may expose a
   Route for the main API, but it exposes no admin Route — any public Route
   only fronts the main port. With auth enabled, the manifest lives only on
   Service port 444, so from a developer workstation
   [port-forward the admin Service port](#port-forwarding-from-a-workstation).

## Handling Failures

A failed fetch is not a dead end. Tell the user which step failed, quote the
error, and use the guidance below.

- **Connection refused, DNS failure, or timeout.** `*.svc` names never
  resolve from a workstation, so this usually means the user needs to
  [port-forward](#port-forwarding-from-a-workstation), not that the server is
  down. If port-forwarding also fails, ask the user to start
  `nemoguardrails server` locally or confirm the in-cluster pod is running.
- **404.** On auth-enabled deployments, first confirm the request went to
  port 444: the port 443 proxy returns 404 for anything outside `/v1/*`. A
  404 on port 444 means the deployed image predates the manifest endpoint.
  Don't use the local checkout's `MANIFEST_PATH` to second-guess this; the
  deployment may be on a different version.
- **401.** The token is missing, malformed, or expired. Ask the user to run
  `oc login` and try again.
- **403.** The token is valid but not authorized. Ask the user to run
  `oc auth can-i create nemoguardrails.trustyai.opendatahub.io/<cr-name> -n <namespace>`.
  If the answer is `no`, they need a role that grants `create` on that CR,
  which is editor-level access; view-only users cannot read the manifest.
  Retrying with the same token will not help, whether or not it goes through
  a port-forward.
- **TLS error (unknown CA or hostname mismatch).** Never disable certificate
  verification. Inside a pod, pass
  `--cacert /var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt`.
  Over a port-forward, the certificate names `<cr-name>.<namespace>.svc`,
  not `localhost`, so use the recipe below. Report any other TLS error
  verbatim.
- **Annotation missing.** The operator is older than the annotation. Check
  the CR's `security.opendatahub.io/enable-auth` annotation and build the URL
  as described above.
- **Port-forwarding not possible, or the user declines.** Fall back to the
  "Key fork changes" section of `CLAUDE.md`, and say explicitly that this is
  a best-effort answer that may be out of date. Never make up fork-specific
  behavior.
- **An expected config is missing from `configs`.** Configs that fail to
  load are silently dropped. Ask the user to search the server logs for
  `Skipping config at <path> in manifest: failed to load`. If `configs` is
  empty, the config directory may not be mounted; the logs will show
  `Rails config path ... is missing`. The other sections are still usable.

### Port-Forwarding From A Workstation

Give these commands to the user instead of running them yourself. Don't run
`oc whoami -t` or print the token, so the credential stays out of the
conversation.

**Auth enabled.** Port-forward the Service's admin port (444), not the main
API port (443) and not the OpenShift Route. On OpenShift, that admin port's
certificate is signed by the service CA, which every namespace publishes in
the `openshift-service-ca.crt` ConfigMap:

```text
oc port-forward svc/<cr-name> 4444:444 -n <namespace>

# in a second terminal
oc get configmap openshift-service-ca.crt -n <namespace> \
  -o jsonpath='{.data.service-ca\.crt}' > /tmp/service-ca.crt
curl --cacert /tmp/service-ca.crt \
  --resolve <cr-name>.<namespace>.svc:4444:127.0.0.1 \
  -H "Authorization: Bearer $(oc whoami -t)" \
  https://<cr-name>.<namespace>.svc:4444/admin/info
```

**Auth disabled:**

```text
oc port-forward svc/<cr-name> 8080:80 -n <namespace>
curl http://localhost:8080/admin/info
```

## Using The Manifest Content

1. Fetch the JSON from `/admin/info`. The section and field names are in the
   payload.
2. To learn what the fields mean in the deployed version, fetch
   `/openapi.json` from the same server and read `CapabilityManifest` under
   `components.schemas`. On auth-enabled deployments, use the admin port and
   the same token. If the server's OpenAPI document is unavailable, read
   `nemoguardrails/server/schemas/manifest.py` instead, and tell the user the
   checkout may not match the deployment.
3. Quote the manifest's field values directly, such as the exact endpoint
   path and behavioral contract text, instead of paraphrasing from memory or
   static docs.

## Related Skills

- Use `guardrails-developer-guide` for upstream/general product-usage
  questions.
- Use `guardrails-developer-create-guardrails` when the user wants to build a
  guardrails configuration rather than ask about this fork's own behavior.
