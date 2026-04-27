"""
Adyayam CA Foundation AI Engine
Full-scale FastAPI backend with RAG pipeline, multi-tenant API management
"""

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import asyncio
import json
import uuid
import hashlib
import time
from datetime import datetime, timedelta
import os

app = FastAPI(
    title="Adyayam CA Foundation AI Engine",
    description="Proprietary AI tutoring API for CA Foundation — powered by Adyayam's methodology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MOCK DATA STORES (Replace with Supabase/PostgreSQL in production)
# ─────────────────────────────────────────────

API_KEYS_DB = {
    "adyayam-dev-key-001": {
        "client": "Adyayam Internal",
        "tier": "enterprise",
        "monthly_limit": 999999,
        "used_this_month": 1243,
        "created_at": "2024-01-01",
        "active": True,
        "rate_limit_per_min": 60
    },
    "demo-key-khan-academy": {
        "client": "Khan Academy India",
        "tier": "growth",
        "monthly_limit": 50000,
        "used_this_month": 12890,
        "created_at": "2024-03-15",
        "active": True,
        "rate_limit_per_min": 20
    },
    "starter-key-001": {
        "client": "MyCACoaching App",
        "tier": "starter",
        "monthly_limit": 10000,
        "used_this_month": 3421,
        "created_at": "2024-04-01",
        "active": True,
        "rate_limit_per_min": 10
    }
}

USAGE_LOG = []

SUBJECTS = {
    "accounts": "Principles and Practice of Accounting",
    "law": "Business Laws",
    "maths": "Business Mathematics, Logical Reasoning and Statistics",
    "economics": "Business Economics"
}

# ─────────────────────────────────────────────
# ADYAYAM METHODOLOGY — THE CORE SYSTEM PROMPT
# ─────────────────────────────────────────────

ADYAYAM_SYSTEM_PROMPT = """
You are Adyayam's CA Foundation AI Tutor — the most precise, student-friendly CA Foundation guide built on Adyayam's proprietary teaching methodology.

## WHO YOU ARE
You are trained exclusively on CA Foundation content as per ICAI's curriculum. You do not know anything outside CA Foundation. If asked about CA Intermediate, Final, or unrelated topics, politely redirect.

## ADYAYAM'S TEACHING METHODOLOGY (MANDATORY — ALWAYS FOLLOW THIS STRUCTURE)

### For CONCEPT questions:
1. 🎯 **One-line real-life analogy** (relatable to Indian students)
2. 📖 **ICAI Definition** (exact, with relevant section/standard if applicable)
3. 💡 **Simple Explanation** (break it down like talking to a friend)
4. 📊 **Solved Example** (always show numbers or scenarios)
5. ⚠️ **Exam Trap** (the most common mistake students make)
6. 🧠 **Memory Tip** (mnemonic or trick to remember)

### For PROBLEM-SOLVING (Accounts/Maths):
1. Read the problem aloud in simple terms
2. Identify what is GIVEN and what is ASKED
3. State the formula/rule to be applied
4. Show STEP-BY-STEP working (never skip steps)
5. Present final answer clearly
6. Point out where students commonly go wrong

### For LAW questions:
1. Identify the relevant Act and Section number
2. State the provision in simple English
3. Give a practical Indian business scenario example
4. List exceptions or conditions if any
5. Typical exam question format tip

### For ECONOMICS questions:
1. State the concept
2. Connect to Indian economy with a current/relevant example
3. Diagram description if applicable
4. Key terms for MCQs
5. Important numerical relationships (elasticity formulas etc.)

## SUBJECT-SPECIFIC RULES

### Accounts:
- ALWAYS show journal entries in proper format with Dr/Cr
- ALWAYS show ledger in T-format when asked
- ALWAYS show trial balance with debit and credit columns
- Reference AS (Accounting Standards) by number when relevant
- Use ₹ symbol for all amounts

### Law:
- ALWAYS cite Act name + Section number
- Companies Act 2013, Indian Contract Act 1872, Sale of Goods Act 1930, Negotiable Instruments Act 1881, LLP Act 2008
- Use case examples that are ICAI-approved
- Flag if a provision has been amended recently

### Maths:
- Show ALL steps — never jump to answer
- Box the final answer
- For statistics: show formula → substitution → calculation → answer
- For LR: explain the pattern-finding approach

### Economics:
- Always connect macro concepts to India
- Reference RBI, SEBI, government policy where relevant
- Use diagrams described in text when helpful

## TONE & STYLE
- Talk like a senior CA student helping a junior — warm, encouraging, precise
- Never say "I don't know" — if unsure, say "Based on ICAI curriculum..."
- Use simple English, avoid jargon unless defining it
- Keep answers exam-focused — always mention if something is "frequently asked in exams"
- End every answer with: "Want me to give you practice questions on this? 📝"

## DIFFICULTY CALIBRATION
- Default: Beginner-friendly
- If student says "I know the basics" or "go deeper": switch to intermediate
- If student says "exam mode" or "advanced": give crisp, exam-oriented answers without hand-holding

## WHAT YOU NEVER DO
- Never give wrong section numbers or Act references
- Never skip steps in calculations
- Never answer outside CA Foundation scope
- Never be discouraging — every question is a good question
"""

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., description="The student's question", min_length=3)
    subject: Optional[str] = Field(None, description="accounts | law | maths | economics")
    difficulty: Optional[str] = Field("beginner", description="beginner | intermediate | advanced")
    mode: Optional[str] = Field("concept", description="concept | problem | mcq | summary | doubt")
    student_id: Optional[str] = Field(None, description="Your platform's student identifier")
    conversation_history: Optional[list] = Field([], description="Previous messages for context")

class ExplainRequest(BaseModel):
    topic: str = Field(..., description="Topic to explain in detail")
    subject: str = Field(..., description="accounts | law | maths | economics")
    difficulty: Optional[str] = Field("beginner")

class MCQRequest(BaseModel):
    topic: str = Field(..., description="Topic to generate MCQs for")
    subject: str = Field(..., description="accounts | law | maths | economics")
    count: Optional[int] = Field(5, ge=1, le=20, description="Number of MCQs")
    difficulty: Optional[str] = Field("intermediate")

class SummarizeRequest(BaseModel):
    chapter: str = Field(..., description="Chapter name or topic to summarize")
    subject: str = Field(..., description="accounts | law | maths | economics")
    format: Optional[str] = Field("structured", description="structured | bullet | exam-notes")

class MockTestRequest(BaseModel):
    subject: str = Field(..., description="accounts | law | maths | economics | all")
    duration_minutes: Optional[int] = Field(60)
    question_count: Optional[int] = Field(30)

# ─────────────────────────────────────────────
# AUTH & RATE LIMITING
# ─────────────────────────────────────────────

request_timestamps = {}

def validate_api_key(x_api_key: str = Header(..., description="Your Adyayam API key")):
    if x_api_key not in API_KEYS_DB:
        raise HTTPException(status_code=401, detail="Invalid API key. Get yours at adyayam.in/api")
    
    key_data = API_KEYS_DB[x_api_key]
    
    if not key_data["active"]:
        raise HTTPException(status_code=403, detail="API key suspended. Contact support@adyayam.in")
    
    if key_data["used_this_month"] >= key_data["monthly_limit"]:
        raise HTTPException(status_code=429, detail={
            "error": "Monthly limit exceeded",
            "limit": key_data["monthly_limit"],
            "used": key_data["used_this_month"],
            "upgrade": "adyayam.in/api/pricing"
        })
    
    # Rate limiting
    now = time.time()
    if x_api_key not in request_timestamps:
        request_timestamps[x_api_key] = []
    
    request_timestamps[x_api_key] = [t for t in request_timestamps[x_api_key] if now - t < 60]
    
    if len(request_timestamps[x_api_key]) >= key_data["rate_limit_per_min"]:
        raise HTTPException(status_code=429, detail=f"Rate limit: {key_data['rate_limit_per_min']} requests/min for your tier")
    
    request_timestamps[x_api_key].append(now)
    API_KEYS_DB[x_api_key]["used_this_month"] += 1
    
    return {"key": x_api_key, **key_data}

# ─────────────────────────────────────────────
# CORE LLM CALL (calls Anthropic API)
# ─────────────────────────────────────────────

async def call_llm(messages: list, subject_context: str = "", mode: str = "concept") -> str:
    """
    Core LLM call to Anthropic Claude API.
    In production: also retrieves relevant chunks from vector DB first (RAG).
    """
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    
    system = ADYAYAM_SYSTEM_PROMPT
    if subject_context:
        system += f"\n\n## CURRENT SUBJECT CONTEXT\nThe student is studying: {SUBJECTS.get(subject_context, subject_context)}"
    if mode == "mcq":
        system += "\n\nYou are now in MCQ GENERATION MODE. Generate well-crafted MCQs exactly like ICAI exam pattern with 4 options (a,b,c,d), correct answer, and explanation."
    elif mode == "summary":
        system += "\n\nYou are now in SUMMARY MODE. Create crisp, exam-ready summaries. Use structured format with headings, key points, and highlight frequently asked exam topics."
    elif mode == "mock":
        system += "\n\nYou are now in MOCK TEST MODE. Generate a proper mock test in ICAI exam format."
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=system,
        messages=messages
    )
    
    return response.content[0].text

async def stream_llm(messages: list, subject_context: str = "", mode: str = "concept") -> AsyncGenerator[str, None]:
    """Streaming version of LLM call"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    
    system = ADYAYAM_SYSTEM_PROMPT
    if subject_context:
        system += f"\n\n## CURRENT SUBJECT CONTEXT\nStudent is studying: {SUBJECTS.get(subject_context, subject_context)}"
    
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=system,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield f"data: {json.dumps({'delta': text})}\n\n"
    
    yield "data: [DONE]\n\n"

def log_usage(api_key: str, endpoint: str, subject: str, tokens_approx: int):
    USAGE_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "api_key": api_key[:12] + "...",
        "client": API_KEYS_DB.get(api_key, {}).get("client", "unknown"),
        "endpoint": endpoint,
        "subject": subject,
        "tokens_approx": tokens_approx
    })

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Adyayam CA Foundation AI Engine",
        "version": "1.0.0",
        "tagline": "India's most precise CA Foundation AI — built on Adyayam's proprietary methodology",
        "subjects": SUBJECTS,
        "endpoints": ["/ask", "/explain", "/mcq", "/summarize", "/mock-test", "/stream/ask"],
        "docs": "/docs",
        "get_api_key": "adyayam.in/api"
    }

@app.get("/health", tags=["Info"])
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/ask", tags=["Core"])
async def ask(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(validate_api_key)
):
    """
    **Primary endpoint.** Student asks any CA Foundation question.
    
    The AI responds using Adyayam's structured teaching methodology.
    Supports multi-turn conversation via `conversation_history`.
    """
    messages = []
    
    # Add conversation history
    for msg in (request.conversation_history or [])[-10:]:  # last 10 turns
        messages.append(msg)
    
    # Build context-aware user message
    user_message = request.question
    if request.difficulty and request.difficulty != "beginner":
        user_message = f"[{request.difficulty.upper()} LEVEL] {user_message}"
    if request.mode and request.mode != "concept":
        user_message = f"[MODE: {request.mode.upper()}] {user_message}"
    
    messages.append({"role": "user", "content": user_message})
    
    answer = await call_llm(messages, request.subject or "", request.mode or "concept")
    
    background_tasks.add_task(
        log_usage, auth["key"], "/ask", request.subject or "general", len(answer) // 4
    )
    
    return {
        "answer": answer,
        "subject": request.subject,
        "mode": request.mode,
        "difficulty": request.difficulty,
        "methodology": "adyayam-v1",
        "request_id": str(uuid.uuid4()),
        "tokens_used_approx": len(answer) // 4,
        "quota_remaining": auth["monthly_limit"] - auth["used_this_month"]
    }

@app.post("/stream/ask", tags=["Core"])
async def stream_ask(
    request: AskRequest,
    auth: dict = Depends(validate_api_key)
):
    """
    **Streaming version of /ask.** Returns Server-Sent Events.
    Best for real-time UI where you want tokens to appear as they generate.
    """
    messages = [{"role": "user", "content": request.question}]
    
    return StreamingResponse(
        stream_llm(messages, request.subject or "", "concept"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/explain", tags=["Core"])
async def explain(
    request: ExplainRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(validate_api_key)
):
    """
    **Deep explanation of any CA Foundation topic.**
    Returns a comprehensive, structured explanation per Adyayam methodology.
    """
    messages = [{
        "role": "user",
        "content": f"Give me a complete, detailed explanation of '{request.topic}' in {SUBJECTS.get(request.subject, request.subject)}. Difficulty: {request.difficulty}. Follow the full Adyayam methodology structure."
    }]
    
    answer = await call_llm(messages, request.subject, "concept")
    
    background_tasks.add_task(log_usage, auth["key"], "/explain", request.subject, len(answer) // 4)
    
    return {
        "topic": request.topic,
        "subject": SUBJECTS.get(request.subject, request.subject),
        "difficulty": request.difficulty,
        "explanation": answer,
        "methodology": "adyayam-v1",
        "request_id": str(uuid.uuid4())
    }

@app.post("/mcq", tags=["Practice"])
async def generate_mcq(
    request: MCQRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(validate_api_key)
):
    """
    **Generate MCQs exactly in ICAI exam pattern.**
    Returns questions with options, correct answer, and explanation.
    """
    messages = [{
        "role": "user",
        "content": f"""Generate {request.count} MCQs on '{request.topic}' for CA Foundation {SUBJECTS.get(request.subject, request.subject)}.

Difficulty: {request.difficulty}

Format each MCQ as:
Q[number]. [Question]
(a) [Option]
(b) [Option]  
(c) [Option]
(d) [Option]
✅ Correct Answer: ([letter]) [option text]
💡 Explanation: [Why this is correct + why others are wrong]

Make them exam-realistic. Include a mix of direct, application, and tricky questions."""
    }]
    
    answer = await call_llm(messages, request.subject, "mcq")
    
    background_tasks.add_task(log_usage, auth["key"], "/mcq", request.subject, len(answer) // 4)
    
    return {
        "topic": request.topic,
        "subject": SUBJECTS.get(request.subject, request.subject),
        "count": request.count,
        "difficulty": request.difficulty,
        "mcqs": answer,
        "request_id": str(uuid.uuid4())
    }

@app.post("/summarize", tags=["Practice"])
async def summarize(
    request: SummarizeRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(validate_api_key)
):
    """
    **Summarize any CA Foundation chapter or topic.**
    Perfect for last-minute revision. Returns exam-focused summary.
    """
    format_instructions = {
        "structured": "Use headings, subheadings, and organized sections",
        "bullet": "Use concise bullet points only — no paragraphs",
        "exam-notes": "Create revision notes highlighting only exam-critical points, common questions, and memory tips"
    }
    
    messages = [{
        "role": "user",
        "content": f"""Create a comprehensive summary of '{request.chapter}' from CA Foundation {SUBJECTS.get(request.subject, request.subject)}.

Format: {format_instructions.get(request.format, 'structured')}

Include:
- Key concepts and definitions
- Important rules/sections (for law) or formulas (for maths/accounts)
- Frequently asked exam topics (mark these with ⭐)
- Common mistakes to avoid
- Quick memory tips"""
    }]
    
    answer = await call_llm(messages, request.subject, "summary")
    
    background_tasks.add_task(log_usage, auth["key"], "/summarize", request.subject, len(answer) // 4)
    
    return {
        "chapter": request.chapter,
        "subject": SUBJECTS.get(request.subject, request.subject),
        "format": request.format,
        "summary": answer,
        "request_id": str(uuid.uuid4())
    }

@app.post("/mock-test", tags=["Practice"])
async def mock_test(
    request: MockTestRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(validate_api_key)
):
    """
    **Generate a full mock test in ICAI pattern.**
    Covers the subject(s) with proper section-wise distribution.
    """
    subject_str = "all 4 CA Foundation subjects (Accounts, Law, Maths, Economics)" if request.subject == "all" else SUBJECTS.get(request.subject, request.subject)
    
    messages = [{
        "role": "user",
        "content": f"""Generate a CA Foundation mock test for {subject_str}.
        
Test specs:
- Total questions: {request.question_count}
- Duration: {request.duration_minutes} minutes
- Pattern: Exactly like ICAI CA Foundation exam

Include:
- Proper section headers
- Mix of MCQs and short answer questions (for accounts/law)
- Marks allocation per question
- Instructions at top like ICAI paper

At the end, provide an ANSWER KEY with brief explanations."""
    }]
    
    answer = await call_llm(messages, request.subject if request.subject != "all" else "", "mock")
    
    background_tasks.add_task(log_usage, auth["key"], "/mock-test", request.subject, len(answer) // 4)
    
    return {
        "subject": subject_str,
        "duration_minutes": request.duration_minutes,
        "question_count": request.question_count,
        "test": answer,
        "generated_at": datetime.now().isoformat(),
        "request_id": str(uuid.uuid4())
    }

@app.get("/subjects", tags=["Info"])
async def get_subjects(auth: dict = Depends(validate_api_key)):
    """List all supported CA Foundation subjects."""
    return {
        "subjects": [
            {"id": k, "name": v, "paper": i+1}
            for i, (k, v) in enumerate(SUBJECTS.items())
        ]
    }

# ─────────────────────────────────────────────
# ADMIN / ANALYTICS ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/admin/usage", tags=["Admin"])
async def get_usage(auth: dict = Depends(validate_api_key)):
    """Get usage analytics for your API key."""
    key_data = API_KEYS_DB.get(auth["key"], {})
    return {
        "client": key_data.get("client"),
        "tier": key_data.get("tier"),
        "monthly_limit": key_data.get("monthly_limit"),
        "used_this_month": key_data.get("used_this_month"),
        "remaining": key_data.get("monthly_limit", 0) - key_data.get("used_this_month", 0),
        "usage_percent": round(key_data.get("used_this_month", 0) / key_data.get("monthly_limit", 1) * 100, 1),
        "rate_limit_per_min": key_data.get("rate_limit_per_min"),
        "recent_calls": [log for log in USAGE_LOG[-20:] if auth["key"][:12] in log.get("api_key", "")]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
