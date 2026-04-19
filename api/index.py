import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging
import hashlib
from datetime import datetime
import requests as http_requests
from urllib.parse import quote

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
    # Use underscores — colons in URL-path endpoints caused silent failures
    return f"umbrellanet_chat_{ip_hash}_{session_id}"


# ===============================
# REDIS HELPERS  (Upstash REST API)
# ===============================

def _run_pipeline(commands: list) -> list | None:
    """
    Execute a list of Redis commands atomically via Upstash pipeline.
    Keys and values are passed as JSON array elements — no URL encoding needed.

    Returns the list of per-command results, or None on failure.
    """
    try:
        resp = http_requests.post(
            f"{UPSTASH_REDIS_REST_URL}/pipeline",
            headers=REDIS_HEADERS,
            json=commands,
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        logging.info(f"Pipeline results: {results}")

        # Surface any per-command errors
        for r in results:
            if isinstance(r, dict) and "error" in r:
                logging.error(f"Pipeline command error: {r['error']}")
                return None

        return results
    except Exception as exc:
        logging.error(f"Pipeline error: {exc}")
        return None


def redis_rpush(key: str, *items) -> bool:
    """
    Append items to a Redis list and reset TTL — all via pipeline so
    the key is always a JSON value, never a URL segment (no colon issues).
    """
    commands = [["RPUSH", key] + [json.dumps(item) for item in items],
                ["EXPIRE", key, CONVERSATION_TTL_SECONDS]]
    result = _run_pipeline(commands)
    return result is not None


def redis_lrange(key: str) -> list:
    """
    Fetch all items from a Redis list via pipeline.
    Returns an empty list if the key doesn't exist or on error.
    """
    result = _run_pipeline([["LRANGE", key, 0, -1]])
    if result is None:
        return []

    raw_list = result[0].get("result", [])
    if not isinstance(raw_list, list):
        logging.warning(f"LRANGE unexpected type: {type(raw_list)} — {raw_list}")
        return []

    turns = []
    for item in raw_list:
        try:
            turns.append(json.loads(item))
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Skipping undecodable item: {item!r} ({e})")
    return turns


def redis_del(key: str) -> bool:
    """Delete a key via pipeline."""
    result = _run_pipeline([["DEL", key]])
    return result is not None


# ===============================
# CONVERSATION HELPERS
# ===============================

# In-memory cache of live chat sessions keyed by redis_key.
_session_cache: dict = {}


def _turns_to_gemini(turns: list) -> list:
    """
    Convert Redis-stored turns into Gemini SDK history format.
    System prompt is prepended in memory — never stored in Redis.
    """
    bootstrap = [
        {"role": "user",  "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Understood. I will follow these instructions."]},
    ]
    real_turns = [{"role": t["role"], "parts": [t["text"]]} for t in turns]
    return bootstrap + real_turns


def get_conversation(redis_key: str):
    """
    Returns a live Gemini ChatSession for the given key.

    Strategy:
      1. Check the in-process cache (fast path — same Gunicorn worker).
      2. Load ONLY real conversation turns from Redis (no system prompt).
      3. Prepend the system prompt bootstrap in memory before handing to SDK.
    """
    # 1. In-process cache
    if redis_key in _session_cache:
        return _session_cache[redis_key]

    # 2. Load real turns from Redis (empty list = brand-new user)
    stored_turns = redis_lrange(redis_key)

    # 3. Build Gemini history: system prompt in memory + real turns from Redis
    gemini_history = _turns_to_gemini(stored_turns)

    chat = model.start_chat(history=gemini_history)
    _session_cache[redis_key] = chat
    return chat


def persist_turn(redis_key: str, user_text: str, model_text: str):
    """
    Atomically append exactly one user turn and one model turn to the
    Redis list for this session. No read-modify-write — just RPUSH.
    The system prompt is never written here.
    """
    redis_rpush(
        redis_key,
        {"role": "user",  "text": user_text},
        {"role": "model", "text": model_text},
    )


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

        logging.info(f"Chat request | key={redis_key} | session={session_id}")

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
