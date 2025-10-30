import os
import json
import pathlib
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import faiss  # type: ignore

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
META_PATH = DATA_DIR / "index" / "meta.json"


class ChatMessage(BaseModel):
    message: str


def load_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "faiss.index"
    if not index_path.exists():
        return None, []
    index = faiss.read_index(str(index_path))
    meta: List[Dict[str, Any]] = []
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return index, meta


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # Uses OpenAI embeddings; expects OPENAI_API_KEY
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = np.array([d.embedding for d in resp.data], dtype="float32")
    return vectors


def retrieve(client: OpenAI, index: faiss.Index, meta: List[Dict[str, Any]], query: str, k: int = 6):
    q_vec = embed_texts(client, [query])
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k)
    I = I[0]
    retrieved = []
    for idx in I:
        if idx < 0 or idx >= len(meta):
            continue
        retrieved.append(meta[idx])
    return retrieved


def build_system_prompt() -> str:
    return (
        "You are Wayne Winton, a friendly, knowledgeable locksmith expert and YouTuber. "
        "Speak in Wayne’s personable, practical tone. Be concise, helpful, and honest. "
        "Ground your answers in the provided video transcript excerpts when relevant. "
        "If you don’t know, say so and suggest where to learn more. Avoid medical or legal claims."
    )


def build_user_prompt(user_message: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for c in contexts:
        title = c.get("title", "Unknown Title")
        url = c.get("url", "")
        chunk = c.get("text", "")
        ctx_lines.append(f"Title: {title}\nURL: {url}\nExcerpt: {chunk}")
    ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context available)"
    return (
        f"Context from George’s videos (may be partial excerpts):\n\n{ctx_block}\n\n"
        f"User: {user_message}\n"
        "Answer as George. Be concise and direct. At the end, add a section titled 'Sources' with 1–3 bullet points in the form 'Title — URL' for the most relevant videos from the context."
    )


def chat_completion(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "600")),
    )
    return resp.choices[0].message.content or ""


load_dotenv()
app = FastAPI(title="George Dynavap Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def root_page(request: Request):
        # Require a simple login cookie before serving the frontend index
        if request.cookies.get("logged_in") != "1":
                # Not logged in -> redirect to login page
                return RedirectResponse(url="/login")

        index_path = WEB_DIR / "index.html"
        if index_path.exists():
                return FileResponse(str(index_path))
        return HTMLResponse("<h1>Wayne Winton Chatbot API</h1><p>Frontend missing.</p>")


_LOGIN_HTML = """
<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>Login</title>
        <style>body{font-family:Arial,Helvetica,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;background:#f6f7fb}form{background:#fff;padding:24px;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,.08)}input{display:block;margin:8px 0;padding:8px;width:280px}</style>
    </head>
    <body>
        <form method="post" action="/login">
            <h2>Login</h2>
            <label>Password</label>
            <input name="password" type="password" placeholder="Enter password" autofocus />
            <button type="submit">Log in</button>
        </form>
    </body>
</html>
"""


@app.get("/login")
def login_page(request: Request):
        return HTMLResponse(_LOGIN_HTML)


@app.post("/login")
async def login(request: Request):
        form = await request.form()
        password = form.get("password", "")
        admin_pw = os.getenv("ADMIN_PASSWORD", "password")
        if password == admin_pw:
                # Success: set a simple cookie and redirect to root
                resp = RedirectResponse(url="/", status_code=303)
                resp.set_cookie("logged_in", "1", httponly=True, max_age=3600)
                return resp

        # Failure: show form again with a small error
        html = _LOGIN_HTML.replace("</form>", "<p style='color:red'>Invalid password</p></form>")
        return HTMLResponse(html, status_code=401)


def _avatar_path() -> pathlib.Path | None:
    # Prefer user-supplied image in project root: goerge.png (as provided) or george.png (common spelling)
    for name in ("goerge.png", "george.png"):
        p = BASE_DIR / name
        if p.exists():
            return p
    return None


@app.get("/avatar")
def get_avatar():
    p = _avatar_path()
    if p is not None:
        return FileResponse(str(p))
    # fallback placeholder in web dir
    placeholder = WEB_DIR / "avatar_placeholder.svg"
    if placeholder.exists():
        return FileResponse(str(placeholder), media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "Avatar not found"})


_index, _meta = load_index()


@app.post("/api/chat")
async def api_chat(msg: ChatMessage, request: Request):
    if _index is None or not _meta:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Index not found. Please run ingestion first.",
            },
        )

    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse(status_code=400, content={"error": "OPENAI_API_KEY is not set."})

    client = OpenAI()
    contexts = retrieve(client, _index, _meta, msg.message, k=int(os.getenv("RETRIEVAL_K", "6")))
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(msg.message, contexts)
    answer = chat_completion(client, system_prompt, user_prompt)

    return {"answer": answer, "sources": contexts}
