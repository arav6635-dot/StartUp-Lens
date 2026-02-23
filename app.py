import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request


load_dotenv()

app = Flask(__name__)



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-1.5-flash").split(",")
    if model.strip()
]
FX_API_URL = "https://open.er-api.com/v6/latest/{base}"
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "INR", "CAD", "AUD", "JPY", "SGD", "AED"]
CACHE_TTL_SECONDS = int(os.getenv("ANALYSIS_CACHE_TTL_SECONDS", "21600"))
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _extract_json(payload: str) -> Dict[str, Any]:
    payload = payload.strip()
    if payload.startswith("{"):
        return json.loads(payload)
    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        raise ValueError("No JSON object found in Gemini response")
    return json.loads(match.group(0))


def _cache_key(university: str, degree: str, international: bool, country: str) -> str:
    return "|".join(
        [
            university.strip().lower(),
            degree.strip().lower(),
            "1" if international else "0",
            country.strip().lower(),
        ]
    )


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = ANALYSIS_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["stored_at"] > CACHE_TTL_SECONDS:
        ANALYSIS_CACHE.pop(key, None)
        return None
    return entry["report"]


def _cache_set(key: str, report: Dict[str, Any]) -> None:
    ANALYSIS_CACHE[key] = {"stored_at": time.time(), "report": report}


def _fetch_rates(base_currency: str) -> Dict[str, float]:
    base = (base_currency or "").upper().strip()
    if not base:
        return {}
    try:
        response = requests.get(FX_API_URL.format(base=base), timeout=12)
        response.raise_for_status()
        payload = response.json()
        if payload.get("result") != "success":
            return {}
        rates = payload.get("rates", {})
        filtered = {base: 1.0}
        for code in SUPPORTED_CURRENCIES:
            if code in rates:
                filtered[code] = float(rates[code])
        return filtered
    except Exception:
        return {base: 1.0}


def _call_gemini(university: str, degree: str, international: bool, country: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    prompt = f"""
You are generating a UniversityLens decision report.
Return exactly one valid JSON object and nothing else.

Input:
- University: {university}
- Degree: {degree}
- International student path: {"yes" if international else "no"}
- Country hint (optional): {country or "not provided"}

Rules:
- Do not use placeholders.
- Build all sections from AI reasoning for this specific university + degree.
- Reddit threads must be realistic short student-style forum comments (no markdown, no bullets).
- Salary values must be annual numbers in local currency of the university country.
- Output must be practical and detailed.
- Keep output indices on 0-100 scale.

Schema (strict):
{{
  "university": "string",
  "degree": "string",
  "country": "string",
  "local_currency": "ISO4217 code",
  "sources": ["string", "string", "string"],
  "objective": {{
    "salary_distribution": [
      {{"band":"P10","amount":0}},
      {{"band":"P25","amount":0}},
      {{"band":"P50","amount":0}},
      {{"band":"P75","amount":0}},
      {{"band":"P90","amount":0}}
    ],
    "top_roles": [
      {{"role":"string","p25":0,"median":0,"p75":0,"hiring_signal":"High|Medium|Low"}}
    ],
    "annual_tuition": 0,
    "annual_living_cost": 0,
    "avg_debt_at_graduation": 0,
    "debt_burden_index": 0,
    "dropout_rate_pct": 0,
    "time_to_employment_months": 0.0,
    "industry_placement": [{{"industry":"string","pct":0}}],
    "visa_success_rate_pct": 0
  }},
  "subjective": {{
    "reddit_threads": [{{"user":"string","upvotes":0,"age":"string","text":"string"}}],
    "burnout_culture_signals": ["string"],
    "professor_reputation_clusters": ["string"],
    "social_life_density": "Low|Moderate|High",
    "hidden_admin_issues": ["string"]
  }},
  "output": {{
    "roi_score": 0,
    "academic_pressure_index": 0,
    "social_isolation_risk": 0,
    "career_outcome_probability": 0,
    "decision_gap": 0,
    "verdict": "ALIGNED|MODERATE GAP|CRITICAL GAP",
    "summary": "string"
  }},
  "insights": {{
    "overall": ["string"],
    "risks": ["string"],
    "opportunities": ["string"]
  }}


}}
""".strip()
    
    models_to_try = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
    last_exc: Optional[Exception] = None

    for model in models_to_try:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={GEMINI_API_KEY}"
            )

            response = requests.post(
                url,
                timeout=60,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            response.raise_for_status()
            body = response.json()

            text = (
                body.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if not text:
                raise RuntimeError("Gemini returned an empty response")

            return _extract_json(text)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_exc = exc
            if status in (404, 429, 500, 502, 503, 504):
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini model available")


def _safe_api_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            return "AI provider quota/rate limit reached. Wait for quota reset or switch to a billed key."
        if status == 404:
            return "AI model endpoint not found. Check GEMINI_MODEL or fallback model names."
        if status == 401 or status == 403:
            return "AI provider authentication failed. Check GEMINI_API_KEY."
        if status == 400:
            return "AI provider rejected the request format."
        if status and status >= 500:
            return "AI provider is currently unavailable. Please retry shortly."
        if status:
            return f"AI provider request failed (HTTP {status}). Check model access and billing/quota."
        return "AI provider request failed."
    if isinstance(exc, requests.RequestException):
        return "Network error while contacting AI provider."
    return "AI analysis failed. Please try again."


def _sanitize_report(raw: Dict[str, Any]) -> Dict[str, Any]:
    objective = raw.get("objective", {})
    subjective = raw.get("subjective", {})
    output = raw.get("output", {})
    insights = raw.get("insights", {})

    salary_distribution = [
        {
            "band": str(row.get("band", "")).strip(),
            "amount": _safe_float(row.get("amount"), 0.0),
        }
        for row in objective.get("salary_distribution", [])
        if isinstance(row, dict)
    ]

    top_roles = [
        {
            "role": str(row.get("role", "")).strip(),
            "p25": _safe_float(row.get("p25"), 0.0),
            "median": _safe_float(row.get("median"), 0.0),
            "p75": _safe_float(row.get("p75"), 0.0),
            "hiring_signal": str(row.get("hiring_signal", "")).strip() or "Medium",
        }
        for row in objective.get("top_roles", [])
        if isinstance(row, dict)
    ]

    industry = [
        {
            "industry": str(row.get("industry", "")).strip(),
            "pct": _safe_int(row.get("pct"), 0),
        }
        for row in objective.get("industry_placement", [])
        if isinstance(row, dict)
    ]

    reddit_threads = [
        {
            "user": str(row.get("user", "anon")).strip() or "anon",
            "upvotes": _safe_int(row.get("upvotes"), 0),
            "age": str(row.get("age", "recent")).strip() or "recent",
            "text": str(row.get("text", "")).strip(),
        }
        for row in subjective.get("reddit_threads", [])
        if isinstance(row, dict)
    ]

    clean = {
        "university": str(raw.get("university", "")).strip(),
        "degree": str(raw.get("degree", "")).strip(),
        "country": str(raw.get("country", "")).strip(),
        "local_currency": str(raw.get("local_currency", "USD")).upper().strip() or "USD",
        "sources": [str(x).strip() for x in raw.get("sources", []) if str(x).strip()],
        "objective": {
            "salary_distribution": salary_distribution,
            "top_roles": top_roles,
            "annual_tuition": _safe_float(objective.get("annual_tuition"), 0.0),
            "annual_living_cost": _safe_float(objective.get("annual_living_cost"), 0.0),
            "avg_debt_at_graduation": _safe_float(objective.get("avg_debt_at_graduation"), 0.0),
            "debt_burden_index": _safe_int(objective.get("debt_burden_index"), 0),
            "dropout_rate_pct": _safe_int(objective.get("dropout_rate_pct"), 0),
            "time_to_employment_months": _safe_float(objective.get("time_to_employment_months"), 0.0),
            "industry_placement": industry,
            "visa_success_rate_pct": (
                None
                if objective.get("visa_success_rate_pct") in (None, "", "N/A")
                else _safe_int(objective.get("visa_success_rate_pct"), 0)
            ),
        },
        "subjective": {
            "reddit_threads": reddit_threads,
            "burnout_culture_signals": [str(x).strip() for x in subjective.get("burnout_culture_signals", []) if str(x).strip()],
            "professor_reputation_clusters": [str(x).strip() for x in subjective.get("professor_reputation_clusters", []) if str(x).strip()],
            "social_life_density": str(subjective.get("social_life_density", "Moderate")).strip() or "Moderate",
            "hidden_admin_issues": [str(x).strip() for x in subjective.get("hidden_admin_issues", []) if str(x).strip()],
        },
        "output": {
            "roi_score": _safe_int(output.get("roi_score"), 0),
            "academic_pressure_index": _safe_int(output.get("academic_pressure_index"), 0),
            "social_isolation_risk": _safe_int(output.get("social_isolation_risk"), 0),
            "career_outcome_probability": _safe_int(output.get("career_outcome_probability"), 0),
            "decision_gap": _safe_int(output.get("decision_gap"), 0),
            "verdict": str(output.get("verdict", "ALIGNED")).strip() or "ALIGNED",
            "summary": str(output.get("summary", "")).strip(),
        },
        "insights": {
            "overall": [str(x).strip() for x in insights.get("overall", []) if str(x).strip()],
            "risks": [str(x).strip() for x in insights.get("risks", []) if str(x).strip()],
            "opportunities": [str(x).strip() for x in insights.get("opportunities", []) if str(x).strip()],
        },
    }

    if clean["output"]["decision_gap"] == 0:
        clean["output"]["decision_gap"] = abs(
            clean["output"]["roi_score"] - clean["output"]["academic_pressure_index"]
        )

    return clean


@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.post("/api/analyze")
def analyze() -> Any:
    payload = request.get_json(silent=True) or {}

    university = str(payload.get("university", "")).strip()
    degree = str(payload.get("degree", "")).strip()
    country = str(payload.get("country", "")).strip()
    international = bool(payload.get("international", False))
    cache_key = _cache_key(university, degree, international, country)

    if not university or not degree:
        return jsonify({"error": "university and degree are required"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "Missing GEMINI_API_KEY. Add it in your environment or .env."}), 503

    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    try:
        raw_report = _call_gemini(university, degree, international, country)
        report = _sanitize_report(raw_report)
    except Exception as exc:
        return jsonify({"error": _safe_api_error(exc)}), 502

    report["fx_rates"] = _fetch_rates(report.get("local_currency", "USD"))
    report["supported_currencies"] = SUPPORTED_CURRENCIES
    _cache_set(cache_key, report)

    return jsonify(report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
