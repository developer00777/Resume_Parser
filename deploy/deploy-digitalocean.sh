#!/usr/bin/env bash
#
# Deploy the resume parser to DigitalOcean App Platform.
#
#   ./deploy/deploy-digitalocean.sh
#
# Uses DigitalOcean Container Registry rather than a GitHub source connection.
# That is deliberate: App Platform's GitHub integration deploys a *branch of a
# repo you can push to*, and this code currently lives on a local branch of a
# repo we only have read access to. Building locally and pushing an image needs
# no GitHub write access at all.
#
# Once the code is on a branch you can push, `deploy/digitalocean-app.yaml`
# describes the GitHub-sourced equivalent and is the better long-term setup —
# it redeploys on push.
#
# Prerequisites
#   brew install doctl && doctl auth init      # needs a DO API token
#   Docker Desktop running
#
set -euo pipefail

APP_NAME="${APP_NAME:-resume-parser}"
REGISTRY_NAME="${REGISTRY_NAME:-}"
# DigitalOcean uses two different region namespaces, and mixing them up gives a
# 422 that names no alternative:
#   App Platform      -> short slugs   (blr, nyc, fra ...)   `doctl apps list-regions`
#   Container Registry -> compute slugs (blr1, nyc3, fra1 ...) `doctl registry options available-regions`
REGION="${REGION:-blr}"                      # App Platform
REGISTRY_REGION="${REGISTRY_REGION:-blr1}"   # Container Registry
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ── Preflight ────────────────────────────────────────────────────────────────
step "Checking prerequisites"

command -v doctl >/dev/null || die "doctl not installed. Run: brew install doctl && doctl auth init"
doctl account get >/dev/null 2>&1 || die "doctl is not authenticated. Run: doctl auth init"
docker info >/dev/null 2>&1 || die "Docker is not running. Start Docker Desktop and retry."

echo "  doctl:  $(doctl version | head -1)"
echo "  account: $(doctl account get --format Email --no-header)"

# A registry is per-account and must already exist; creating one silently would
# commit you to a monthly cost you did not ask for.
if [[ -z "$REGISTRY_NAME" ]]; then
    REGISTRY_NAME="$(doctl registry get --format Name --no-header 2>/dev/null || true)"
fi
[[ -n "$REGISTRY_NAME" ]] || die \
"No container registry found. Create one first (it has a monthly cost):

     doctl registry create champ-registry --region $REGISTRY_REGION

   Note the region slug: the registry wants '$REGISTRY_REGION', not the App
   Platform form '$REGION'. Full list: doctl registry options available-regions
   Then re-run this script."
echo "  registry: $REGISTRY_NAME"

IMAGE="registry.digitalocean.com/${REGISTRY_NAME}/${APP_NAME}"

# ── Secrets ──────────────────────────────────────────────────────────────────
# Read interactively and never echoed, written, or committed. If they are
# already exported, they are used as-is so this can run unattended in CI.
step "Collecting secrets"

prompt_secret() {
    local var="$1" label="$2"
    if [[ -z "${!var:-}" ]]; then
        read -rsp "  ${label}: " "$var"
        echo
    else
        echo "  ${label}: (from environment)"
    fi
    [[ -n "${!var:-}" ]] || die "$label is required"
}

prompt_secret OPENROUTER_API_KEY "OpenRouter API key"
prompt_secret API_KEY            "API_KEY for this service (must match the X-API-Key header in Salesforce)"
prompt_secret SF_CLIENT_ID       "Salesforce Consumer Key"
prompt_secret SF_CLIENT_SECRET   "Salesforce Consumer Secret"

# Client Credentials requires the org's My Domain host. login.salesforce.com and
# test.salesforce.com both reject it with "request not supported on this domain",
# so there is no safe default to fall back to — ask.
if [[ -z "${SF_LOGIN_URL:-}" ]]; then
    echo
    echo "  Salesforce My Domain URL (NOT login/test.salesforce.com)."
    echo "  Find it in Setup -> My Domain, or copy it from the address bar after logging in."
    echo "  e.g. https://acme--demosbx.sandbox.my.salesforce.com"
    read -rp "  My Domain URL: " SF_LOGIN_URL
fi
SF_LOGIN_URL="${SF_LOGIN_URL%/}"
[[ "$SF_LOGIN_URL" == *my.salesforce.com* ]] || die \
"'$SF_LOGIN_URL' is not a My Domain URL. The Client Credentials Flow is
   rejected on login.salesforce.com and test.salesforce.com with
   'invalid_grant: request not supported on this domain'.
   Use the host from Setup -> My Domain instead."
echo "  Salesforce login URL: $SF_LOGIN_URL"

# ── Build and push ───────────────────────────────────────────────────────────
step "Building image ${IMAGE}:${IMAGE_TAG}"

# linux/amd64 explicitly: App Platform runs amd64, and an image built on an
# Apple Silicon Mac defaults to arm64 and fails to start with no useful error.
docker build --platform linux/amd64 -f docker/Dockerfile -t "${IMAGE}:${IMAGE_TAG}" .
docker tag "${IMAGE}:${IMAGE_TAG}" "${IMAGE}:latest"

step "Pushing to the registry"
doctl registry login
docker push "${IMAGE}:${IMAGE_TAG}"
docker push "${IMAGE}:latest"

# ── App spec ─────────────────────────────────────────────────────────────────
step "Writing the app spec"

SPEC="$(mktemp -t resume-parser-spec)"
trap 'rm -f "$SPEC"' EXIT

cat > "$SPEC" <<SPEC_EOF
name: ${APP_NAME}
region: ${REGION}
services:
  - name: api
    image:
      registry_type: DOCR
      repository: ${APP_NAME}
      tag: ${IMAGE_TAG}
    http_port: 8000
    # One instance, one worker: the async /parse/job endpoints keep their job
    # store in process memory. See docs/REMAINING_WORK.md item 3.
    instance_count: 1
    instance_size_slug: basic-xs
    health_check:
      http_path: /health
      initial_delay_seconds: 20
      period_seconds: 30
      timeout_seconds: 10
      failure_threshold: 3
    envs:
      - { key: OPENROUTER_API_KEY, scope: RUN_TIME, type: SECRET, value: "${OPENROUTER_API_KEY}" }
      - { key: API_KEY,            scope: RUN_TIME, type: SECRET, value: "${API_KEY}" }
      - { key: SF_CLIENT_ID,       scope: RUN_TIME, type: SECRET, value: "${SF_CLIENT_ID}" }
      - { key: SF_CLIENT_SECRET,   scope: RUN_TIME, type: SECRET, value: "${SF_CLIENT_SECRET}" }
      - { key: PUBLIC_BASE_URL,        scope: RUN_TIME, value: "\${APP_URL}" }
      - { key: SF_LOGIN_URL,           scope: RUN_TIME, value: "${SF_LOGIN_URL}" }
      - { key: SF_API_VERSION,         scope: RUN_TIME, value: "59.0" }
      - { key: SF_DESCRIBE_TTL_MINUTES, scope: RUN_TIME, value: "60" }
      - { key: OPENROUTER_MODEL,       scope: RUN_TIME, value: "openai/gpt-4o-mini" }
      - { key: OPENROUTER_OCR_MODEL,   scope: RUN_TIME, value: "openai/gpt-4o-mini" }
      - { key: WEB_CONCURRENCY,        scope: RUN_TIME, value: "1" }
      - { key: LOG_LEVEL,              scope: RUN_TIME, value: "INFO" }
SPEC_EOF

# ── Create or update ─────────────────────────────────────────────────────────
EXISTING="$(doctl apps list --format ID,Spec.Name --no-header 2>/dev/null \
            | awk -v n="$APP_NAME" '$2==n {print $1}' | head -1)"

if [[ -n "$EXISTING" ]]; then
    step "Updating existing app ($EXISTING)"
    doctl apps update "$EXISTING" --spec "$SPEC" --wait
    APP_ID="$EXISTING"
else
    step "Creating app"
    APP_ID="$(doctl apps create --spec "$SPEC" --format ID --no-header --wait)"
fi

APP_URL="$(doctl apps get "$APP_ID" --format DefaultIngress --no-header)"

# ── Verify ───────────────────────────────────────────────────────────────────
step "Verifying"

echo "  waiting for the health check..."
for _ in $(seq 1 30); do
    if curl -fsS --max-time 10 "${APP_URL}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 10
done

echo "  /health           -> $(curl -fsS --max-time 15 "${APP_URL}/health" || echo 'NOT RESPONDING')"
echo "  /describe/status  -> $(curl -fsS --max-time 20 -H "X-API-Key: ${API_KEY}" \
        "${APP_URL}/api/v1/salesforce/describe/status" || echo 'NOT RESPONDING')"

cat <<DONE

────────────────────────────────────────────────────────────────────────────
Deployed.

  App ID : ${APP_ID}
  URL    : ${APP_URL}
  Logs   : doctl apps logs ${APP_ID} --follow

If "loaded" is false in describe/status above, the parser reached Salesforce but
could not read the schema — check SF_CLIENT_ID/SF_CLIENT_SECRET, and that the
Connected App has Client Credentials Flow enabled with a Run-As user.

NEXT, IN SALESFORCE (docs/SALESFORCE_INTEGRATION.md):

  1. Named Credential 'Resume_Parser_API' -> URL ${APP_URL}
  2. External Credential custom header  X-API-Key = the API_KEY you entered
  3. Permission Set -> External Credential Principal Access -> add the principal
  4. Dry run before anything writes:
       curl -X POST "${APP_URL}/api/v1/salesforce/parse-and-update?record_id=<id>&dry_run=true" \\
         -H "X-API-Key: <API_KEY>"

NOTE ON EGRESS IP: App Platform has no stable outbound address. If the
integration user's profile has Login IP Ranges, tick "Relax IP restrictions" on
the Connected App, or redeploy to a Droplet using
deploy/docker-compose.prod.yml instead.
────────────────────────────────────────────────────────────────────────────
DONE
