#!/usr/bin/env bash
# Full end-to-end demo of the Semantic Docs API.
# Usage: bash examples/demo.sh [BASE_URL]
# Default BASE_URL: http://localhost:8000

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0

# ── helpers ──────────────────────────────────────────────────────────────────

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n"  "$*"; }
divider(){ printf "\n%s\n" "────────────────────────────────────────"; }

check() {
  local label="$1" actual="$2" expected="$3"
  if echo "$actual" | grep -q "$expected"; then
    green "  PASS  $label"
    PASS=$((PASS + 1))
  else
    red   "  FAIL  $label"
    red   "        expected to find: $expected"
    red   "        in response:      $actual"
    FAIL=$((FAIL + 1))
  fi
}

request() {
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -s -X "$method" "$BASE_URL$path" \
      -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -s -X "$method" "$BASE_URL$path"
  fi
}

# ── 0. cleanup — remove sources this demo creates if they already exist ───────

divider
bold "0. Cleanup (ensures a clean run)"

for source in fastapi-guide docker-guide; do
  result=$(request DELETE "/sources/$source" 2>/dev/null || true)
  if echo "$result" | grep -q "deleted_chunks"; then
    echo "   Removed leftover: $source"
  fi
done

# ── 1. health ─────────────────────────────────────────────────────────────────

divider
bold "1. Health check"

response=$(request GET /health)
echo "   $response"
check "status is ok" "$response" '"status":"ok"'

# ── 2. index documents ────────────────────────────────────────────────────────

divider
bold "2. Indexing documents"

response=$(request POST /index '{
  "source_id": "fastapi-guide",
  "title": "FastAPI Guide",
  "content": "FastAPI is a modern web framework for building APIs with Python. It uses standard Python type hints and provides automatic OpenAPI documentation. FastAPI supports async and sync endpoints. It uses Pydantic for data validation and serialization. Performance is on par with NodeJS and Go frameworks."
}')
echo "   $response"
check "fastapi-guide indexed" "$response" '"status":"ok"'
check "chunks_indexed >= 1"   "$response" 'chunks_indexed'

response=$(request POST /index '{
  "source_id": "docker-guide",
  "title": "Docker Guide",
  "content": "Docker packages applications into containers. A container includes the code, runtime, libraries, and configuration. docker compose up starts all services defined in docker-compose.yml. Containers are isolated from the host but can share ports and volumes. Docker images are built from a Dockerfile."
}')
echo "   $response"
check "docker-guide indexed" "$response" '"status":"ok"'

# ── 3. list sources ───────────────────────────────────────────────────────────

divider
bold "3. Listing indexed sources"

response=$(request GET /sources)
echo "   $response"
check "fastapi-guide present" "$response" 'fastapi-guide'
check "docker-guide present"  "$response" 'docker-guide'

# ── 4. ask questions ──────────────────────────────────────────────────────────

divider
bold "4. Asking questions"

printf "\n   Question: What does FastAPI use for data validation?\n"
response=$(request POST /ask '{"question": "What does FastAPI use for data validation?", "top_k": 3}')
echo "   Answer:   $(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['answer'][:200])")"
check "answer references fastapi-guide" "$response" 'fastapi-guide'

printf "\n   Question: How do I start all services with Docker?\n"
response=$(request POST /ask '{"question": "How do I start all services with Docker?", "top_k": 3}')
echo "   Answer:   $(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['answer'][:200])")"
check "answer references docker-guide" "$response" 'docker-guide'

printf "\n   Question filtered to fastapi-guide only:\n"
response=$(request POST /ask '{"question": "How does FastAPI handle performance?", "top_k": 3, "source_ids": ["fastapi-guide"]}')
check "filtered search works" "$response" 'fastapi-guide'
check "docker-guide not in filtered result" "$(echo "$response" | grep -v docker-guide)" 'fastapi-guide'

# ── 5. ask with no matching content ───────────────────────────────────────────

divider
bold "5. Fallback — question with no relevant content"

response=$(request POST /ask '{"question": "What is the capital of France?"}')
echo "   $response"
check "fallback message returned" "$response" 'No relevant documentation found'

# ── 6. idempotent re-index ────────────────────────────────────────────────────

divider
bold "6. Re-indexing same source_id (idempotency)"

response=$(request POST /index '{
  "source_id": "fastapi-guide",
  "title": "FastAPI Guide v2",
  "content": "FastAPI is a modern web framework for building APIs with Python. Updated content."
}')
echo "   $response"
check "re-index returns ok" "$response" '"status":"ok"'

response=$(request GET /sources)
check "fastapi-guide still present after re-index" "$response" 'fastapi-guide'

# ── 7. delete a source ────────────────────────────────────────────────────────

divider
bold "7. Deleting a source"

response=$(request DELETE /sources/docker-guide)
echo "   $response"
check "delete returns deleted_chunks" "$response" 'deleted_chunks'

response=$(request GET /sources)
check "docker-guide is gone"    "$response" 'fastapi-guide'
# docker-guide must not appear in sources anymore
if echo "$response" | grep -q '"source_id":"docker-guide"'; then
  red "  FAIL  docker-guide still present after delete"
  FAIL=$((FAIL + 1))
else
  green "  PASS  docker-guide removed from sources"
  PASS=$((PASS + 1))
fi

response=$(request DELETE /sources/docker-guide)
check "second delete returns 404" "$response" '"detail"'

# ── summary ───────────────────────────────────────────────────────────────────

divider
TOTAL=$((PASS + FAIL))
bold "Results: $PASS/$TOTAL passed"

if [ "$FAIL" -eq 0 ]; then
  green "All checks passed."
  exit 0
else
  red "$FAIL check(s) failed."
  exit 1
fi
