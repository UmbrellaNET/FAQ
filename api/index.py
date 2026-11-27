import hashlib
import hmac 
import logging
import os 
from datetime import datetime, time, timedelta, timezone
from functools import wraps

import google.generativeai as genai
import requests as http_requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

# ===============================
# CONFIGURATION
# ===============================

logging.basicConfig(level=logging.INFO)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable not set")
    return value


GEN_API_KEY = _require_env("GEN_API")
MODEL_NAME = "gemini-2.5-flash"

# Supabase (secret key, server side only)
SUPABASE_URL = _require_env("SUPABASE_URL").rstrip("/")
SUPABASE_SERVICE_KEY = _require_env("SUPABASE_SERVICE_KEY")

# Secret used to hash IP addresses so they cannot be reversed by brute force
IP_HASH_SALT = _require_env("IP_HASH_SALT")

# Dashboard sign-in: Firebase project and the emails allowed to use admin routes
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "umbrellalabs-faq")
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}

# Sites allowed to call this API from a browser (the website and the dashboard)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# How much recent history is sent to Gemini (the old Redis setup used a 24 hour expiry)
CONTEXT_HOURS = int(os.environ.get("CONTEXT_HOURS", "24"))
CONTEXT_MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 4000

if not ADMIN_EMAILS:
    logging.warning("ADMIN_EMAILS is empty: every admin route will be refused")
if not ALLOWED_ORIGINS:
    logging.warning("ALLOWED_ORIGINS is empty: browsers will be blocked by CORS")

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
    {"category": "HARM_CATEGORY_HARASSMENT",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",      "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
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

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# ===============================
# SUPABASE HELPERS  (PostgREST over HTTPS)
# ===============================


def _sb(method: str, path: str, *, params=None, body=None, prefer: str | None = None):
    """
    Call Supabase's REST API. The secret key goes in the apikey header only:
    sb_secret_ keys are not JWTs, so sending them as a bearer token fails.
    """
    headers = {"apikey": SUPABASE_SERVICE_KEY, "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    resp = http_requests.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=headers,
        params=params,
        json=body,
        timeout=8,
    )
    resp.raise_for_status()
    return resp


def db_load_context(ip_hash: str, session_id: str) -> list:
    """Recent messages for one conversation, oldest first, starting with a user message."""
    since = (datetime.now(timezone.utc) - timedelta(hours=CONTEXT_HOURS)).isoformat()
    try:
        resp = _sb("GET", "chat_messages", params={
            "ip_hash": f"eq.{ip_hash}",
            "session_id": f"eq.{session_id}",
            "created_at": f"gte.{since}",
            "select": "role,text:content",
            "order": "id.desc",
            "limit": str(CONTEXT_MAX_MESSAGES),
        })
        rows = list(reversed(resp.json()))
    except Exception:
        logging.exception("Could not load conversation history")
        return []

    # Gemini history must start with a user turn
    while rows and rows[0]["role"] != "user":
        rows.pop(0)
    return rows


def db_save_turn(ip_hash: str, session_id: str, user_text: str, model_text: str) -> bool:
    """Append one user message and one model message."""
    try:
        _sb("POST", "chat_messages", prefer="return=minimal", body=[
            {"ip_hash": ip_hash, "session_id": session_id, "role": "user",  "content": user_text},
            {"ip_hash": ip_hash, "session_id": session_id, "role": "model", "content": model_text},
        ])
        return True
    except Exception:
        logging.exception("Could not save conversation turn")
        return False


def db_clear(ip_hash: str, session_id: str) -> None:
    _sb("DELETE", "chat_messages", prefer="return=minimal", params={
        "ip_hash": f"eq.{ip_hash}",
        "session_id": f"eq.{session_id}",
    })


# ===============================
# USER IDENTITY HELPERS
# ===============================


def get_client_ip() -> str:
    """Real client IP, honouring common reverse-proxy headers."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()
    return request.remote_addr or "unknown"


def hash_ip(ip: str) -> str:
    """Salted hash, so raw addresses are never stored and cannot be brute forced."""
    return hmac.new(IP_HASH_SALT.encode(), ip.encode(), hashlib.sha256).hexdigest()[:16]


def _clean(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


# ===============================
# ADMIN AUTHENTICATION  (Firebase ID token)
# ===============================

_google_request = google_requests.Request()


def require_admin(fn):
    """Allow only signed-in dashboard users whose verified email is in ADMIN_EMAILS."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            claims = google_id_token.verify_firebase_token(
                header[7:], _google_request, audience=FIREBASE_PROJECT_ID
            )
        except Exception:
            return jsonify({"error": "Unauthorized"}), 401
        if not claims:
            return jsonify({"error": "Unauthorized"}), 401

        email = (claims.get("email") or "").lower()
        if not claims.get("email_verified") or email not in ADMIN_EMAILS:
            return jsonify({"error": "Forbidden"}), 403
        return fn(*args, **kwargs)
    return wrapper


def _db_failed(exc: Exception):
    logging.error("Database error: %s", exc)
    return jsonify({"error": "Database error", "status": "error"}), 502


# ===============================
# PUBLIC ROUTES
# ===============================


@app.route("/")
def home():
    return jsonify({"status": "AI Agent API is running", "model": MODEL_NAME})


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.get_json(silent=True) or {}
        message = _clean(data.get("message"), MAX_MESSAGE_CHARS + 1)
        session_id = _clean(data.get("session_id", "default"), 100) or "default"

        if not message:
            return jsonify({"error": "Message is required"}), 400
        if len(message) > MAX_MESSAGE_CHARS:
            return jsonify({"error": f"Message is too long (max {MAX_MESSAGE_CHARS} characters)"}), 400

        ip_hash = hash_ip(get_client_ip())
        logging.info("Chat request | ip=%s | session=%s", ip_hash, session_id)

        # The system prompt is added in memory and never stored
        history = [
            {"role": "user",  "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I will follow these instructions."]},
        ] + [{"role": r["role"], "parts": [r["text"]]} for r in db_load_context(ip_hash, session_id)]

        convo = model.start_chat(history=history)
        response = convo.send_message(message)
        answer = response.text

        db_save_turn(ip_hash, session_id, message, answer)

        needs_human = "unable_to_solve_query" in answer
        clean_answer = answer.replace("unable_to_solve_query", "").strip()

        return jsonify({
            "response": clean_answer,
            "needs_human": needs_human,
            "session_id": session_id,
            "status": "success",
        })

    except Exception as e:
        logging.error("Chat error: %s", e)
        return jsonify({"error": "Internal server error", "status": "error"}), 500


@app.route("/api/clear", methods=["POST"])
def clear_history():
    try:
        data = request.get_json(silent=True) or {}
        session_id = _clean(data.get("session_id", "default"), 100) or "default"
        db_clear(hash_ip(get_client_ip()), session_id)
        return jsonify({"message": f"Conversation cleared for session {session_id}", "status": "success"})
    except Exception as e:
        logging.error("Clear error: %s", e)
        return jsonify({"error": "Internal server error", "status": "error"}), 500


@app.route("/api/missed-request", methods=["POST", "OPTIONS"])
def missed_request():
    """Called by the chat widget when nobody was online to take a human chat."""
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    row = {
        "session_id": _clean(data.get("session_id"), 100) or None,
        "name":       _clean(data.get("name"), 200) or None,
        "email":      _clean(data.get("email"), 200) or None,
        "phone":      _clean(data.get("phone"), 50) or None,
        "message":    _clean(data.get("message"), 2000) or None,
    }
    if not row["email"] and not row["phone"]:
        return jsonify({"error": "An email or phone number is required"}), 400

    try:
        _sb("POST", "missed_requests", prefer="return=minimal", body=row)
    except Exception as e:
        return _db_failed(e)
    return jsonify({"status": "success"})


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        _sb("GET", "chat_messages", params={"select": "id", "limit": "1"})
        db_ok = True
    except Exception:
        db_ok = False
    return jsonify({
        "status": "healthy" if db_ok else "degraded",
        "model": MODEL_NAME,
        "database": "ok" if db_ok else "unreachable",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200 if db_ok else 503


# ===============================
# ADMIN ROUTES  (dashboard)
# ===============================


@app.route("/api/admin/sessions", methods=["GET"])
@require_admin
def admin_sessions():
    try:
        rows = _sb("GET", "session_summaries", params={
            "order": "last_at.desc",
            "limit": "300",
        }).json()
    except Exception as e:
        return _db_failed(e)
    return jsonify({"sessions": rows})


@app.route("/api/admin/messages", methods=["GET"])
@require_admin
def admin_messages():
    ip_hash = _clean(request.args.get("ip_hash"), 64)
    session_id = _clean(request.args.get("session_id"), 100)
    if not ip_hash or not session_id:
        return jsonify({"error": "ip_hash and session_id are required"}), 400
    try:
        rows = _sb("GET", "chat_messages", params={
            "ip_hash": f"eq.{ip_hash}",
            "session_id": f"eq.{session_id}",
            "select": "role,text:content,created_at",
            "order": "id.asc",
            "limit": "1000",
        }).json()
    except Exception as e:
        return _db_failed(e)
    return jsonify({"messages": rows})


@app.route("/api/admin/analytics", methods=["GET"])
@require_admin
def admin_analytics():
    today = datetime.now(timezone.utc).date()
    try:
        start_day = datetime.strptime(request.args["start"], "%Y-%m-%d").date() if "start" in request.args \
            else today - timedelta(days=182)
        end_day = datetime.strptime(request.args["end"], "%Y-%m-%d").date() if "end" in request.args else today
    except ValueError:
        return jsonify({"error": "Dates must look like 2026-11-27"}), 400

    p_start = datetime.combine(start_day, time.min, tzinfo=timezone.utc).isoformat()
    p_end = datetime.combine(end_day, time.max, tzinfo=timezone.utc).isoformat()

    try:
        s = _sb("POST", "rpc/analytics_summary", body={"p_start": p_start, "p_end": p_end}).json()
    except Exception as e:
        return _db_failed(e)

    s["avg_messages"] = round(s["total_messages"] / s["total_sessions"], 1) if s["total_sessions"] else 0
    s["retention"] = round(100 * s["returning_users"] / s["total_users"]) if s["total_users"] else 0
    s["response_rate"] = round(100 * s["bot_messages"] / s["user_messages"]) if s["user_messages"] else 0
    return jsonify(s)


@app.route("/api/admin/missed", methods=["GET"])
@require_admin
def admin_missed():
    try:
        rows = _sb("GET", "missed_requests", params={
            "order": "created_at.desc",
            "limit": "300",
        }).json()
    except Exception as e:
        return _db_failed(e)
    return jsonify({"requests": rows, "unseen": sum(1 for r in rows if not r.get("seen_at"))})


@app.route("/api/admin/missed/count", methods=["GET"])
@require_admin
def admin_missed_count():
    try:
        resp = _sb("GET", "missed_requests", prefer="count=exact", params={
            "seen_at": "is.null",
            "select": "id",
            "limit": "1",
        })
        # Content-Range looks like "0-0/5" or "*/0"
        unseen = int(resp.headers.get("Content-Range", "*/0").split("/")[-1])
    except Exception as e:
        return _db_failed(e)
    return jsonify({"unseen": unseen})


@app.route("/api/admin/missed/<int:missed_id>/seen", methods=["POST"])
@require_admin
def admin_missed_seen(missed_id: int):
    try:
        _sb("PATCH", "missed_requests", prefer="return=minimal",
            params={"id": f"eq.{missed_id}"},
            body={"seen_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return _db_failed(e)
    return jsonify({"status": "success"})


@app.route("/api/admin/missed/seen-all", methods=["POST"])
@require_admin
def admin_missed_seen_all():
    try:
        _sb("PATCH", "missed_requests", prefer="return=minimal",
            params={"seen_at": "is.null"},
            body={"seen_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return _db_failed(e)
    return jsonify({"status": "success"})


# ===============================
# LOCAL DEV ONLY
# ===============================

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=8000)
