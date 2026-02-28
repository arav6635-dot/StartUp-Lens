# StartupLens: Destroy My Startup

A Flask app that generates roast or praise for startup landing pages.

## What it does

- Accepts a startup URL from the homepage.
- Checks Neon Postgres cache by normalized domain + mode (`roast` / `praise`).
- If already generated, reuses cached result instantly.
- If new, fetches landing-page text + external domain enrichment context and calls Gemini.
- Stores generated result and shows it on a detailed result page.
- Shows recent results on homepage.

## Stack

- Python + Flask
- Neon Postgres (via `psycopg`)
- Gemini API (HTTP call to Generative Language API)
- Vanilla HTML/CSS templates

## Setup

1. Create venv and install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and fill values:

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
DATABASE_URL=postgresql://...?...sslmode=require
DOMAIN_ENRICHMENT_CACHE_HOURS=168
```

3. Run:

```bash
python app.py
```

4. Open:

`http://127.0.0.1:5001`

## Neon table

Tables are created automatically on startup:

`roasts(normalized_url + mode UNIQUE, roast_text, request_count, timestamps, ...)`
`domain_enrichment(domain PK, summary, sources, updated_at)`

This guarantees repeated URL+mode requests do not trigger Gemini again, and domain enrichment is reused.
