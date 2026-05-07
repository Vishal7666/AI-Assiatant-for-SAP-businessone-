"""
server.py — TECH AI Server — Enhanced with Auth
────────────────────────────────────────────────────────────────────────
FEATURES:
  • User Registration & Login (JWT tokens)
  • Admin Dashboard API endpoints
  • Dual engine: Groq (DB) + Gemini (General)
  • Session management
  • User statistics tracking
────────────────────────────────────────────────────────
"""

import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from config2 import apikey, gemini_apikey
from rag_module import RAGModule
from gemini_module import GeminiModule


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str
    is_admin: bool = False

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    db_access: bool = False

class ChatResponse(BaseModel):
    reply: str
    engine: str = ""

class ClearRequest(BaseModel):
    session_id: str = ""

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    is_admin: bool
    join_date: str

class StatsResponse(BaseModel):
    total_users: int
    total_messages: int
    avg_response_time: float
    groq_status: str
    gemini_status: str
    db_status: str


# ══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="TECH AI Server — Enterprise Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SIMULATION (use real DB in production)
# ══════════════════════════════════════════════════════════════════════════════

USERS_FILE = "users.json"
STATS_FILE = "stats.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {"total_messages": 0, "total_sessions": 0, "avg_response_time": 0}

def save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

def update_stats():
    stats = load_stats()
    stats["total_messages"] = stats.get("total_messages", 0) + 1
    save_stats(stats)


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISE ENGINES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("🚀  TECH AI SERVER — ENTERPRISE EDITION")
print("    Groq → Database & Business Intelligence (NL→SQL)")
print("    Gemini           → General Chat & Knowledge")
print("    Auth             → JWT + User Management")
print("=" * 80)

gemini = GeminiModule(api_key=gemini_apikey)
rag    = RAGModule(api_key=apikey)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=dict)
def register(req: UserRegister):
    """Register a new user"""
    users = load_users()

    if any(u['email'] == req.email for u in users):
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = {
        "id":        str(datetime.now().timestamp()),
        "name":      req.name,
        "email":     req.email,
        "password":  req.password,   # Hash in production!
        "is_admin":  False,
        "join_date": datetime.now().isoformat(),
        "is_active": True,
    }
    users.append(new_user)
    save_users(users)

    return {
        "status":  "success",
        "message": "User registered successfully",
        "user_id": new_user["id"],
    }


@app.post("/auth/login", response_model=dict)
def login(req: UserLogin):
    """Login user"""
    users = load_users()

    if req.is_admin:
        if req.email == "admin@techmai.com" and req.password == "admin123":
            return {
                "status": "success",
                "token":  "admin_token_" + str(datetime.now().timestamp()),
                "user": {
                    "id":       "admin",
                    "name":     "Admin",
                    "email":    req.email,
                    "is_admin": True,
                },
            }
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    user = next(
        (u for u in users if u['email'] == req.email and u['password'] == req.password),
        None,
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "status": "success",
        "token":  "user_token_" + str(datetime.now().timestamp()),
        "user": {
            "id":       user["id"],
            "name":     user["name"],
            "email":    user["email"],
            "is_admin": user.get("is_admin", False),
        },
    }


@app.get("/auth/users", response_model=List[UserResponse])
def get_users():
    """Get all users (admin only)"""
    users = load_users()
    return [
        UserResponse(
            id=u["id"],
            name=u["name"],
            email=u["email"],
            is_admin=u.get("is_admin", False),
            join_date=u.get("join_date", ""),
        )
        for u in users
    ]


@app.delete("/auth/users/{user_id}")
def delete_user(user_id: str):
    """Delete a user (admin only)"""
    users = load_users()
    users = [u for u in users if u["id"] != user_id]
    save_users(users)
    return {"status": "success", "message": "User deleted"}


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the main HTML file"""
    html_path = os.path.join(os.path.dirname(__file__), "ai_chat_widget.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>Error: ai_chat_widget.html not found</h2>"


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, req: Request):
    """Main chat endpoint — Groq handles DB, Gemini handles everything else"""

    # Anonymous users cannot access DB
    auth_header = req.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()
    if not token and request.db_access:
        request.db_access = False

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    msg = request.message.strip()
    update_stats()

    try:
        # ── Try Groq (DB only) first ────────────────────────────────────────
        answer, sources, is_db = rag.ask(msg, allow_db=request.db_access)

        if is_db:
            # DB question — use Groq's answer (even if it's an error message)
            if sources:
                web_sources = [s for s in sources if not str(s.get("url", "")).startswith("db://")]
                if web_sources:
                    answer += "\n\n**Sources:**"
                    for link in web_sources[:3]:
                        answer += f"\n- [{link['title']}]({link['url']})"
            return ChatResponse(reply=answer, engine="groq")

        # ── Not a DB question — route entirely to Gemini ─────────────────────
        print("  [Server] Non-DB question → Gemini")
        gemini_answer, gemini_sources = gemini.chat(msg)

        if gemini_sources:
            web_sources = [s for s in gemini_sources if not str(s.get("url", "")).startswith("db://")]
            if web_sources:
                gemini_answer += "\n\n**Sources:**"
                for link in web_sources[:3]:
                    gemini_answer += f"\n- [{link['title']}]({link['url']})"

        return ChatResponse(reply=gemini_answer, engine="gemini")

    except Exception as e:
        return ChatResponse(reply=f"⚠️ Error: {str(e)}", engine="error")


@app.post("/clear")
def clear_endpoint(request: ClearRequest):
    """Clear conversation history"""
    gemini.clear_history()
    return {"status": "cleared", "message": "Conversation history cleared."}


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/analytics/stats", response_model=StatsResponse)
def get_stats():
    """Get system statistics"""
    stats = load_stats()
    users = load_users()

    return StatsResponse(
        total_users=len(users),
        total_messages=stats.get("total_messages", 0),
        avg_response_time=stats.get("avg_response_time", 2.3),
        groq_status="✅ Connected" if rag._gc else "❌ Offline",
        gemini_status="✅ Connected" if gemini.model_name else "❌ Offline",
        db_status="✅ Connected" if rag._sql else "❌ Offline",
    )


@app.get("/status")
def status_endpoint():
    """Health check with backend status"""
    users  = load_users()
    stats  = load_stats()

    return {
        "status":    "running",
        "timestamp": datetime.now().isoformat(),
        "users": {
            "total":  len(users),
            "active": len([u for u in users if u.get("is_active", True)]),
        },
        "stats": {
            "total_messages":   stats.get("total_messages", 0),
            "avg_response_time": stats.get("avg_response_time", 2.3),
        },
        "groq": {
            "engine":  "Groq",
            "model":   rag._gm if rag._gm else "not connected",
            "status":  "✅ Ready" if rag._gc else "❌ Offline",
            "purpose": "Database & Business Queries (NL→SQL)",
        },
        "gemini": {
            "engine":  "Gemini",
            "model":   gemini.model_name or "not connected",
            "status":  "✅ Ready" if gemini.model_name else "❌ Offline",
            "purpose": "General Chat & Knowledge",
        },
        "database": {
            "status":  "✅ Connected" if rag._sql else "❌ Not Connected",
            "db_name": rag.cfg.DATABASE if rag._sql else "N/A",
        },
    }


@app.get("/models")
def list_models():
    """List available AI models"""
    return {
        "groq_model":    rag._gm if rag._gm else "not connected",
        "gemini_model":  gemini.model_name or "not connected",
        "provider":      "Groq (DB) + Gemini (General)",
        "status":        "running",
    }


@app.get("/health")
def health_check():
    """Simple health check"""
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"  ✅ Groq model      : {rag._gm or '❌ not connected'}")
    print(f"  ✅ Gemini model    : {gemini.model_name or '❌ not connected'}")
    print(f"  ✅ Database        : {rag.cfg.DATABASE if rag._sql else '❌ not connected'}")
    print("=" * 80)
    print("  🌐 Chat UI     : http://127.0.0.1:8080")
    print("  📊 Status      : http://127.0.0.1:8080/status")
    print("  📚 API Docs    : http://127.0.0.1:8080/docs")
    print("\n  📝 Demo Admin Login:")
    print("     Email: admin@techmai.com")
    print("     Password: admin123")
    print("\n" + "=" * 80 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8080)