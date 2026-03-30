import google.generativeai as genai
from flask import Flask, request, jsonify 
from flask_cors import CORS 
import os
import logging
from datetime import datetime

# ===============================
# CONFIGURATION
# ===============================

logging.basicConfig(level=logging.INFO)

GEN_API_KEY = os.environ.get("GEN_API")
MODEL_NAME = "gemini-2.5-flash"

if not GEN_API_KEY:
    raise RuntimeError("GEN_API environment variable not set")

genai.configure(api_key=GEN_API_KEY)

# Load system instructions (STATIC, SAFE)
from .instructions import instructions as SYSTEM_PROMPT

# ===============================
# GEMINI MODEL (SAFE INIT)
# ===============================

generation_config = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# ===============================
# FLASK APP
# ===============================

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://iamf-a-e.github.io"]
    }
})

# ===============================
# CONVERSATION STORAGE
# (Replace with Redis in production)
# ===============================

conversations = {}

def get_conversation(session_id: str):
    """
    Creates a chat session with SYSTEM_PROMPT injected
    safely via chat history (SDK compatible)
    """
    if session_id not in conversations:
        conversations[session_id] = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [SYSTEM_PROMPT]
                }
            ]
        )
    return conversations[session_id]


# ===============================
# ROUTES
# ===============================

@app.route("/")
def home():
    return jsonify({
        "status": "AI Agent API is running",
        "model": MODEL_NAME
    })

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        convo = get_conversation(session_id)
        response = convo.send_message(message)

        answer = response.text if hasattr(response, "text") else str(response)
   

        # 🔍 HUMAN ESCALATION TOKEN
        needs_human = "unable_to_solve_query" in answer

        # CLEAN RESPONSE
        clean_answer = answer.replace("unable_to_solve_query", "").strip()

        return jsonify({
            "response": clean_answer,
            "needs_human": needs_human,
            "session_id": session_id,
            "status": "success"
        })

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "error"
        }), 500

@app.route("/api/clear", methods=["POST"])
def clear_history():
    try:
        session_id = (request.json or {}).get("session_id", "default")
        conversations.pop(session_id, None)

        return jsonify({
            "message": f"Conversation cleared for session {session_id}",
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        test_model = genai.GenerativeModel(MODEL_NAME)
        test_model.generate_content("Health check")

        return jsonify({
            "status": "healthy",
            "model": MODEL_NAME,
            "timestamp": str(datetime.now())
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# ===============================
# LOCAL DEV ONLY
# ===============================

if __name__ == "__main__":
    app.run(debug=True, port=8000)
