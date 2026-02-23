# UniversityLens

UniversityLens is a Flask-based web app to evaluate a **university + degree** beyond rankings and marketing.

It combines:
- Objective outcomes (salary distribution, debt, dropout, placement, visa outcomes)
- Subjective signals (Reddit-style student complaints, burnout culture, admin friction)
- AI-synthesized decision metrics (ROI, pressure, isolation risk, career probability)

## Core Features

- Gemini-powered analysis pipeline (no hardcoded report content)
- Reddit-thread style perception panel
- Detailed salary analysis:
  - Percentile distribution (P10/P25/P50/P75/P90)
  - Top roles with P25/P50/P75 and hiring signal
- Currency conversion:
  - Local currency
  - USD
  - Additional selectable currencies (from FX rates)
- Export report as PNG
- Themed in-page error handling (429/auth/network/general)
- Safe backend error messages (no API key leakage)
- In-memory caching to reduce repeat API calls and rate-limit pressure

## Tech Stack

- Backend: Python, Flask, Requests, python-dotenv
- Frontend: HTML/CSS/vanilla JS
- AI: Google Gemini API
- Screenshot export: `html2canvas`

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```


## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` in project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-2.0-flash,gemini-1.5-flash
ANALYSIS_CACHE_TTL_SECONDS=21600
```

4. Run app:

```bash
python app.py
```

5. Open:

`http://127.0.0.1:5001`

## Environment Variables

- `GEMINI_API_KEY` (required): Gemini API key
- `GEMINI_MODEL` (optional): primary model (default: `gemini-2.0-flash`)
- `GEMINI_FALLBACK_MODELS` (optional): comma-separated fallback models
- `ANALYSIS_CACHE_TTL_SECONDS` (optional): in-memory report cache TTL

## API

### `POST /api/analyze`

Request body:

```json
{
  "university": "University Name",
  "degree": "Degree Program",
  "country": "Optional country hint",
  "international": true
}
```

Returns a structured report including:
- `objective`
- `subjective`
- `output`
- `insights`
- `local_currency`
- `fx_rates`
- `supported_currencies`

## Error Handling

- 429 errors are expected on free-tier quotas during peak usage
- UI shows a styled error panel with actionable message + retry
- Backend sanitizes provider errors to prevent leaking secrets

## Security Notes

- Never commit `.env`
- Rotate keys immediately if exposed
- This project already ignores `.env` via `.gitignore`

## Troubleshooting

- **Port already in use**: change port in `app.py` or kill process on `5001`
- **429 rate limit**: wait and retry, use fallback models, or move to billed quota
- **Model not found (404)**: verify `GEMINI_MODEL` / `GEMINI_FALLBACK_MODELS`
- **Auth error (401/403)**: verify API key and provider access

## Future Improvements

- Add persistent caching (Redis/SQLite)
- Add real external data connectors for outcomes validation
- Add report history and comparisons across universities
- Add authentication and usage analytics
