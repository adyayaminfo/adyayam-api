# Adyayam CA Foundation AI Engine
## Complete Setup Guide

---

## What You've Built

A full-scale proprietary AI tutoring API for CA Foundation with:
- **RAG pipeline** — your PDFs become a searchable knowledge base
- **Adyayam methodology** — your teaching style encoded in every response  
- **Multi-tenant API** — sell access to competitors with per-key billing
- **7 endpoints** — ask, stream, explain, MCQ, summarize, mock test, analytics

---

## Prerequisites

Install these before anything:
- Python 3.11+
- pip

---

## Step 1: Clone & Install

```bash
cd adyayam/backend
pip install -r requirements.txt
```

---

## Step 2: Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Where to get it | Required |
|---|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com | YES |
| `OPENAI_API_KEY` | platform.openai.com | YES (for embeddings) |
| `PINECONE_API_KEY` | pinecone.io | Production only |
| `LLAMAPARSE_API_KEY` | cloud.llamaindex.ai | Recommended |

---

## Step 3: Add Your PDFs

Create a `pdfs/` folder and organize your CA Foundation materials:

```
backend/
  pdfs/
    accounts/
      basic_accounting_icai.pdf
      final_accounts.pdf
      depreciation.pdf
    law/
      indian_contract_act.pdf
      companies_act_basics.pdf
      negotiable_instruments.pdf
    maths/
      ratio_proportion.pdf
      statistics.pdf
      logical_reasoning.pdf
    economics/
      demand_supply.pdf
      national_income.pdf
      money_banking.pdf
```

**Tips for PDFs:**
- Use ICAI study materials for accuracy
- Text-based PDFs work best (not scanned images)
- If you have scanned PDFs, run them through Google Drive first:
  - Upload to Google Drive → Right-click → Open with Google Docs → Download as PDF
  - This adds an OCR text layer

---

## Step 4: Build the Knowledge Base

```bash
# Run the ingestion pipeline (builds your vector database)
python rag_pipeline.py

# Check what was indexed
python rag_pipeline.py stats
```

This will:
1. Parse all your PDFs
2. Split into smart chunks (respecting journal entries, sections, etc.)
3. Generate embeddings
4. Store in ChromaDB (local) or Pinecone (production)

**Expected output:**
```
🚀 Adyayam RAG Pipeline — Starting ingestion
========================================
📚 Found 12 PDFs to process

📄 Processing: basic_accounting_icai.pdf
  → Extracted 84,231 characters
  → Detected subject: accounts
  → Created 127 chunks
  → Generating embeddings...
  → Storing in vector DB...
  ✅ Done! 127 chunks indexed

...

✅ Ingestion complete!
   Total PDFs processed: 12
   Total chunks stored: 1,847
```

---

## Step 5: Start the API

```bash
# Development
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Test it's working:
```bash
curl http://localhost:8000
```

---

## Step 6: Test All Endpoints

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: adyayam-dev-key-001" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Trial Balance and why is it prepared?",
    "subject": "accounts",
    "difficulty": "beginner"
  }'
```

### Generate MCQs
```bash
curl -X POST http://localhost:8000/mcq \
  -H "X-API-Key: adyayam-dev-key-001" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Indian Contract Act - Essential elements of a valid contract",
    "subject": "law",
    "count": 5,
    "difficulty": "intermediate"
  }'
```

### Summarize a chapter
```bash
curl -X POST http://localhost:8000/summarize \
  -H "X-API-Key: adyayam-dev-key-001" \
  -H "Content-Type: application/json" \
  -d '{
    "chapter": "Depreciation",
    "subject": "accounts",
    "format": "exam-notes"
  }'
```

### Generate a mock test
```bash
curl -X POST http://localhost:8000/mock-test \
  -H "X-API-Key: adyayam-dev-key-001" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "accounts",
    "duration_minutes": 60,
    "question_count": 30
  }'
```

### Check your usage
```bash
curl http://localhost:8000/admin/usage \
  -H "X-API-Key: adyayam-dev-key-001"
```

### Interactive API docs
Open in browser: http://localhost:8000/docs

---

## Step 7: Add Real API Key Management

The current `main.py` uses a hardcoded dictionary for API keys.
Replace with Supabase for production:

### Install Supabase
```bash
pip install supabase
```

### Create this table in Supabase:
```sql
CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key TEXT UNIQUE NOT NULL,
  client_name TEXT NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('starter', 'growth', 'enterprise')),
  monthly_limit INTEGER NOT NULL,
  used_this_month INTEGER DEFAULT 0,
  rate_limit_per_min INTEGER NOT NULL,
  active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT NOW(),
  billing_email TEXT
);

-- Create usage log table
CREATE TABLE usage_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  api_key TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  subject TEXT,
  tokens_used INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Reset usage on 1st of each month (set up as a cron job in Supabase)
-- UPDATE api_keys SET used_this_month = 0;
```

### Update main.py to use Supabase:
```python
from supabase import create_client

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

def validate_api_key(x_api_key: str = Header(...)):
    result = supabase.table("api_keys").select("*").eq("key", x_api_key).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return result.data[0]
```

---

## Step 8: Deploy to Production

### Option A: Railway (Easiest — recommended to start)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Railway auto-detects FastAPI and deploys with a public URL.
Set your environment variables in Railway dashboard.

### Option B: Render

1. Push code to GitHub
2. Create new Web Service on render.com
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables

### Option C: AWS (For Enterprise scale)

```bash
# Using AWS App Runner — simplest AWS option
aws apprunner create-service \
  --service-name adyayam-api \
  --source-configuration '{"ImageRepository": {...}}'
```

---

## Step 9: Set Up API Documentation for Clients

Your API docs are auto-generated at `/docs` (Swagger UI).

For a branded developer portal, use **Mintlify**:

1. Sign up at mintlify.com (free)
2. Create `docs/mint.json`:

```json
{
  "name": "Adyayam API",
  "logo": { "dark": "/logo/dark.svg", "light": "/logo/light.svg" },
  "api": {
    "baseUrl": "https://api.adyayam.in",
    "auth": { "method": "key", "name": "X-API-Key" }
  },
  "navigation": [
    {
      "group": "Getting Started",
      "pages": ["introduction", "authentication", "quickstart"]
    },
    {
      "group": "Endpoints",
      "pages": ["api-reference/ask", "api-reference/mcq", "api-reference/summarize"]
    }
  ]
}
```

---

## Step 10: Sell the API

### Pricing setup
Use **Stripe** for API billing:

```bash
pip install stripe
```

Create products in Stripe dashboard:
- Starter: ₹499/month (10,000 queries)
- Growth: ₹1,499/month (50,000 queries)
- Enterprise: Custom

Hook Stripe webhooks to your Supabase API keys table to auto-provision keys on payment.

### Who to approach first
1. **CA coaching app founders** on LinkedIn — search "CA coaching app India"
2. **EdTech communities** — iSpirt, Headstart Network
3. **ICAI registered coaching institutes** looking to go digital
4. **Study material publishers** — Snow White, ICAI itself

### Your pitch
> "Adyayam has built India's only AI tutor trained exclusively for CA Foundation,
> with subject-specific methodology that mirrors how top CA faculties teach.
> Instead of building this from scratch (6+ months, ₹50L+), 
> integrate our API in a weekend. Starting at ₹499/month."

---

## Architecture Summary

```
Student Query
     ↓
[API Gateway] ← validate API key, rate limit, log usage
     ↓
[RAG Retriever] ← fetch relevant chunks from your PDFs
     ↓
[Claude API] ← Adyayam system prompt + retrieved context + query
     ↓
[Structured Response] → streamed or batched back to client
     ↓
[Usage Logger] → Supabase → billing
```

---

## Upgrading the Methodology

Your system prompt in `main.py` under `ADYAYAM_SYSTEM_PROMPT` is your core IP.
Iterate on it as you learn what students struggle with:

- Add subject-specific mnemonics
- Add "frequently asked in ICAI exams" flags
- Add Hinglish explanations for certain concepts
- Add different teaching modes (exam warrior vs. conceptual deep-dive)

Each iteration makes your API more valuable and harder to replicate.

---

## Cost Estimates (Monthly)

| Item | Cost |
|---|---|
| Claude API (50K queries × avg 800 tokens) | ~$40/month |
| Pinecone (1M vectors) | $70/month |
| Railway hosting | $5/month |
| OpenAI embeddings (one-time ingestion) | ~$2 |
| **Total infra cost** | **~$115/month (~₹9,500)** |

At Growth tier pricing (₹1,499/month × 3 clients = ₹4,497):
- Break even at ~7 clients
- At 20 clients: ₹30,000 revenue − ₹15,000 infra = ₹15,000 margin

Scale to enterprise contracts and the margins improve dramatically.
