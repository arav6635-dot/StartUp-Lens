# StartUp Lens

Simple web app to check startup website and generate:
- Roast
- Praise

Built with Python Flask, Gemini API, and Neon Postgres.

## How it works

- User puts website URL.
- App collects website data + some web context.
- Gemini gives output (roast or praise).
- Result is saved in Neon DB.
- If same URL is used again, cached result is shown fast.

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
DATABASE_URL=your_neon_db_url
DOMAIN_ENRICHMENT_CACHE_HOURS=168
```

Run app:

```bash
python app.py
```

Open in browser:
`http://127.0.0.1:5001`
