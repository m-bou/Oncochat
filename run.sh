#!/usr/bin/env bash
# OncoChat task runner. Usage: ./run.sh <command> [args...]   (./run.sh help)
# Windows equivalent: run.ps1 (same commands). Keep both in sync.
#
# Setup: init (create .env from .env.example if missing; also run by `up` and the devcontainer postCreate)
# Host commands (need Docker): up, down, logs, status
# Dev commands (need uv, e.g. inside the devcontainer): dev, test, lint, fmt
# eval runs with uv when available, otherwise inside the running app container.
set -euo pipefail
cd "$(dirname "$0")"

APP_URL="http://localhost:8000"

die()     { echo "error: $*" >&2; exit 1; }
has()     { command -v "$1" >/dev/null 2>&1; }
need_uv() {
  has uv || die "'uv' not found. Open the project in the VS Code devcontainer, or install uv: https://docs.astral.sh/uv/"
}
compose() {
  has docker || die "docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/"
  docker compose "$@"
}

# Create .env from the template, never overwriting an existing one. When run as root (devcontainer
# postCreate), hand the file to the owner of the template so it stays editable from a Linux host.
ensure_env() {
  [ -f .env ] && return 0
  [ -f .env.example ] || die ".env.example not found"
  cp .env.example .env
  if [ "$(id -u)" = "0" ]; then
    chown "$(stat -c '%u:%g' .env.example 2>/dev/null || stat -f '%u:%g' .env.example)" .env 2>/dev/null || true
  fi
  echo "created .env from .env.example (edit it to change models, threads, Ollama host...)"
}

usage() {
  cat <<EOF
Usage: ./run.sh <command> [args...]

  init          Create .env from .env.example if it does not exist (never overwrites)
  up            Build and start everything in the background (first run downloads ~7 GB of models)
  down          Stop the stack (models and history are kept in Docker volumes)
  logs          Follow app and Ollama logs
  status        Show container status and the app health check

  dev           Run the API with auto-reload on :8000 (devcontainer or local uv)
  test [args]   Offline test suite with a fake LLM (args go to pytest, e.g. -k guard)
  lint          ruff check
  fmt           ruff format + ruff check --fix
  eval [args]   Live evaluation against Ollama (args go to eval/run_eval.py, e.g. --tier fast --repeat 3)

  help          Show this message
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  init)
    ensure_env
    ;;
  up)
    ensure_env
    compose up --build -d "$@"
    echo "Starting: $APP_URL  (first run: the model download can take several minutes; ./run.sh logs)"
    ;;
  down)
    compose down "$@"
    ;;
  logs)
    compose logs -f app ollama "$@"
    ;;
  status)
    compose ps
    if has curl; then
      echo
      curl -fsS "$APP_URL/api/health" && echo || echo "app not reachable at $APP_URL"
    fi
    ;;
  dev)
    need_uv
    exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload "$@"
    ;;
  test)
    need_uv
    exec uv run pytest -q "$@"
    ;;
  lint)
    need_uv
    exec uv run ruff check . "$@"
    ;;
  fmt)
    need_uv
    uv run ruff format .
    exec uv run ruff check . --fix
    ;;
  eval)
    if has uv; then
      exec uv run python -m eval.run_eval "$@"
    else
      echo "uv not found: running the eval inside the app container"
      compose exec app python -m eval.run_eval "$@"
      # bring the JSON reports back to the host (never overwrite existing files)
      tmp="$(mktemp -d)"
      compose cp app:/app/eval/results/. "$tmp" >/dev/null
      cp -n "$tmp"/*.json eval/results/ 2>/dev/null && echo "reports copied to eval/results/"
      rm -rf "$tmp"
    fi
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    die "unknown command '$cmd'"
    ;;
esac
