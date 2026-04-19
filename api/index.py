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

def redis_del(key: str) -> bool:
    """Delete a key from Redis using the direct REST endpoint."""
    try:
        resp = http_requests.delete(
            f"{UPSTASH_REDIS_REST_URL}/del/{key}",
            headers=REDIS_HEADERS,
            timeout=5,
        )
        resp.raise_for_status()
        logging.debug(f"Redis DEL response for '{key}': {resp.json()}")
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


def _turns_to_gemini(turns: list) -> list:
    """
    Convert our Redis-stored turns (plain dicts with role/text)
    into the Gemini SDK history format.
    The system-prompt bootstrap turns are prepended in memory here
    and are never stored in Redis.
    """
    # System prompt bootstrap — in memory only, never persisted
    bootstrap = [
        {"role": "user",  "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Understood. I will follow these instructions."]},
    ]
    real_turns = [
        {"role": t["role"], "parts": [t["text"]]}
        for t in turns
    ]
    return bootstrap + real_turns


def redis_rpush(key: str, *items) -> bool:
    """
    Append items to a Redis list and reset TTL.

    Sends a single RPUSH with all values in one command, then EXPIRE.
    Upstash REST supports multi-value RPUSH:
      POST /rpush/key  body: [val1, val2, ...]
    Then a separate EXPIRE call to reset TTL.
    """
    try:
        serialised = [json.dumps(item) for item in items]

        # Single RPUSH with all values — avoids multiple round-trips
        # and type-conflict issues from separate pipeline commands
        push_resp = http_requests.post(
            f"{UPSTASH_REDIS_REST_URL}/rpush/{key}",
            headers=REDIS_HEADERS,
            json=serialised,
            timeout=5,
        )
        push_resp.raise_for_status()
        push_result = push_resp.json()
        logging.info(f"Redis RPUSH result for '{key}': {push_result}")

        if "error" in push_result:
            logging.error(f"Redis RPUSH error in response: {push_result['error']}")
            return False

        # Reset TTL
        expire_resp = http_requests.post(
            f"{UPSTASH_REDIS_REST_URL}/expire/{key}/{CONVERSATION_TTL_SECONDS}",
            headers=REDIS_HEADERS,
            timeout=5,
        )
        expire_resp.raise_for_status()
        logging.debug(f"Redis EXPIRE result for '{key}': {expire_resp.json()}")

        return True
    except Exception as exc:
        logging.error(f"Redis RPUSH error for key '{key}': {exc}")
        return False


def redis_lrange(key: str) -> list:
    """
    Fetch all items from a Redis list using a direct GET call
    (not pipeline) so the response structure is unambiguous:
      {"result": ["item1", "item2", ...]}

    Returns an empty list if the key doesn't exist or on error.
    """
    try:
        # Use the direct command endpoint: GET /lrange/key/0/-1
        resp = http_requests.get(
            f"{UPSTASH_REDIS_REST_URL}/lrange/{key}/0/-1",
            headers=REDIS_HEADERS,
            timeout=5,
        )
        resp.raise_for_status()

        data = resp.json()
        logging.info(f"Redis LRANGE response for '{key}': {data}")

        raw_list = data.get("result", [])

        if not isinstance(raw_list, list):
            logging.warning(f"Redis LRANGE unexpected result type: {type(raw_list)} — {raw_list}")
            return []

        turns = []
        for item in raw_list:
            try:
                turns.append(json.loads(item))
            except (json.JSONDecodeError, TypeError) as parse_err:
                logging.warning(f"Skipping undecodable Redis list item: {item!r} ({parse_err})")

        return turns
    except Exception as exc:
        logging.error(f"Redis LRANGE error for key '{key}': {exc}")
        return []


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
