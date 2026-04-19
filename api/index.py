import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging
import hashlib
from datetime import datetime
import requests as http_requests

# ===============================
# CONFIGURATION
# ===============================

logging.basicConfig(level=logging.INFO)

GEN_API_KEY = os.environ.get("GEN_API")
MODEL_NAME = "gemini-2.5-flash"

if not GEN_API_KEY:
    raise RuntimeError("GEN_API environment variable not set")

# ── Upstash Redis credentials ──────────────────────────────────
UPSTASH_REDIS_REST_URL   = os.environ.get(
    "UPSTASH_REDIS_REST_URL"
)
UPSTASH_REDIS_REST_TOKEN = os.environ.get(
    "UPSTASH_REDIS_REST_TOKEN"
)

REDIS_HEADERS = {
    "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
    "Content-Type":  "application/json",
}

# How long (seconds) to keep a conversation in Redis before it expires.
# 24 hours by default — change to suit your needs.
CONVERSATION_TTL_SECONDS = 86_400

genai.configure(api_key=GEN_API_KEY)

# Load system instructions (STATIC, SAFE)
from .instructions import instructions as SYSTEM_PROMPT

# ===============================
# GEMINI MODEL
# ===============================

generation_config = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# ===============================
# FLASK APP
# ===============================

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://UmbrellaNET.github.io"]
    }
})

# ===============================
# USER IDENTITY HELPERS
# ===============================

def get_client_ip() -> str:
    """
    Returns the real client IP, honouring common reverse-proxy headers.
    Falls back to the direct socket address if no proxy header is present.
    """
    # Prefer X-Forwarded-For (first entry = real client when behind a proxy/CDN)
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()

    # Fly.io / Railway / Render often use this header
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()

    return request.remote_addr or "unknown"


def build_redis_key(ip: str, session_id: str) -> str:
    """
    Builds a namespaced Redis key that is unique per (IP, session).
    The IP is hashed so raw addresses are never stored in Redis keys.
    """
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
    return f"umbrellanet:chat:{ip_hash}:{session_id}"


# ===============================
# REDIS HELPERS  (Upstash REST API)
# ===============================

def redis_get(key: str):
    """Fetch a JSON-encoded value from Redis. Returns None if not found."""
    try:
        resp = http_requests.get(
            f"{UPSTASH_REDIS_REST_URL}/get/{key}",
            headers=REDIS_HEADERS,
            timeout=5,
        )
        resp.raise_for_status()
        result = resp.json().get("result")
        if result is None:
            return None
        return json.loads(result)
    except Exception as exc:
        logging.error(f"Redis GET error for key '{key}': {exc}")
        return None


def redis_set(key: str, value, ttl: int = CONVERSATION_TTL_SECONDS) -> bool:
    """
    Store a JSON-encoded value in Redis with an expiry.

    Upstash REST API format:
        POST /set/<key>/<value>?EX=<seconds>
    The value must be URL-encoded as a path segment — we use the
    pipeline (array) endpoint instead to avoid encoding headaches
    with complex JSON payloads:
        POST /pipeline  body: [["SET", key, value, "EX", ttl]]
    """
    try:
        serialised = json.dumps(value)
        pipeline   = [["SET", key, serialised, "EX", ttl]]
        resp = http_requests.post(
            f"{UPSTASH_REDIS_REST_URL}/pipeline",
            headers=REDIS_HEADERS,
            json=pipeline,
            timeout=5,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logging.error(f"Redis SET error for key '{key}': {exc}")
        return False


def redis_del(key: str) -> bool:
    """Delete a key from Redis using the pipeline endpoint."""
    try:
        pipeline = [["DEL", key]]
        resp = http_requests.post(
            f"{UPSTASH_REDIS_REST_URL}/pipeline",
            headers=REDIS_HEADERS,
            json=pipeline,
            timeout=5,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logging.error(f"Redis DEL error for key '{key}': {exc}")
        return False


# ===============================
# CONVERSATION HELPERS
# ===============================

# In-memory cache of live chat sessions keyed by redis_key.
# This avoids re-serialising / re-creating the Gemini chat object on every
# request within the same process lifetime.
_session_cache: dict = {}


def _history_to_gemini(history: list) -> list:
    """Convert our serialisable history format to the SDK's expected structure."""
    return [
        {"role": turn["role"], "parts": [turn["text"]]}
        for turn in history
    ]


def get_conversation(redis_key: str):
    """
    Returns a live Gemini ChatSession for the given key.

    Strategy:
      1. Check the in-process cache (fast path — same Gunicorn worker).
      2. Load history from Redis and reconstruct the session (cross-worker / restart safe).
      3. Brand-new session — inject the system prompt as the first user turn.
    """
    # 1. In-process cache
    if redis_key in _session_cache:
        return _session_cache[redis_key]

    # 2. Restore from Redis
    stored_history = redis_get(redis_key)

    if stored_history:
        gemini_history = _history_to_gemini(stored_history)
    else:
        # 3. New session — prime with system prompt
        stored_history = [
            {"role": "user",  "text": SYSTEM_PROMPT},
            # Gemini SDK requires the history to alternate user/model.
            # Provide a minimal model acknowledgement so the SDK is happy.
            {"role": "model", "text": "Understood. I will follow these instructions."},
        ]
        gemini_history = _history_to_gemini(stored_history)
        redis_set(redis_key, stored_history)

    chat = model.start_chat(history=gemini_history)
    _session_cache[redis_key] = chat
    return chat


def persist_turn(redis_key: str, user_text: str, model_text: str):
    """
    Append the latest user + model turn to the Redis history and
    reset the TTL so active conversations don't expire mid-session.

    Guards against a corrupt/unexpected Redis value by always ensuring
    `history` is a list before appending.
    """
    raw = redis_get(redis_key)

    # Defensive: if Redis returned something that isn't a list (e.g. a dict
    # from a previous broken write), start fresh rather than crashing.
    if isinstance(raw, list):
        history = raw
    else:
        if raw is not None:
            logging.warning(
                f"Unexpected Redis value type for key '{redis_key}': "
                f"{type(raw).__name__}. Resetting history."
            )
        history = []

    history.append({"role": "user",  "text": user_text})
    history.append({"role": "model", "text": model_text})
    redis_set(redis_key, history, ttl=CONVERSATION_TTL_SECONDS)


# ===============================
# ROUTES
# ===============================

@app.route("/")
def home():
    return jsonify({
        "status": "AI Agent API is running",
        "model":  MODEL_NAME,
    })


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data       = request.json or {}
        message    = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        client_ip = get_client_ip()
        redis_key = build_redis_key(client_ip, session_id)

        logging.info(f"Chat request | ip_hash={redis_key.split(':')[2]} | session={session_id}")

        convo    = get_conversation(redis_key)
        response = convo.send_message(message)
        answer   = response.text if hasattr(response, "text") else str(response)

        # Persist to Redis immediately after a successful reply
        persist_turn(redis_key, message, answer)

        # 🔍 HUMAN ESCALATION TOKEN
        needs_human = "unable_to_solve_query" in answer

        # CLEAN RESPONSE
        clean_answer = answer.replace("unable_to_solve_query", "").strip()

        return jsonify({
            "response":    clean_answer,
            "needs_human": needs_human,
            "session_id":  session_id,
            "status":      "success",
        })

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({
            "error":  "Internal server error",
            "status": "error",
        }), 500


@app.route("/api/clear", methods=["POST"])
def clear_history():
    try:
        data       = request.json or {}
        session_id = data.get("session_id", "default")
        client_ip  = get_client_ip()
        redis_key  = build_redis_key(client_ip, session_id)

        # Remove from Redis
        redis_del(redis_key)

        # Remove from in-process cache so the next request starts fresh
        _session_cache.pop(redis_key, None)

        return jsonify({
            "message": f"Conversation cleared for session {session_id}",
            "status":  "success",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        # Check Gemini
        test_model = genai.GenerativeModel(MODEL_NAME)
        test_model.generate_content("Health check")

        # Check Redis (simple ping via GET on a dummy key)
        redis_ping = http_requests.get(
            f"{UPSTASH_REDIS_REST_URL}/get/__ping__",
            headers=REDIS_HEADERS,
            timeout=3,
        )
        redis_ok = redis_ping.status_code == 200

        return jsonify({
            "status":     "healthy" if redis_ok else "degraded",
            "model":      MODEL_NAME,
            "redis":      "ok" if redis_ok else "unreachable",
            "timestamp":  str(datetime.now()),
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error":  str(e),
        }), 500


# ===============================
# LOCAL DEV ONLY
# ===============================

if __name__ == "__main__":
    app.run(debug=True, port=8000)
