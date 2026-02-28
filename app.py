import os
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import psycopg
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for


load_dotenv()

app = Flask(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
DOMAIN_ENRICHMENT_CACHE_HOURS = int(
    os.getenv("DOMAIN_ENRICHMENT_CACHE_HOURS", "168")
)
_db_ready = False


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")
    return psycopg.connect(DATABASE_URL, autocommit=True)


def init_db() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS roasts (
                id BIGSERIAL PRIMARY KEY,
                normalized_url TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'roast',
                slug TEXT,
                original_url TEXT NOT NULL,
                display_domain TEXT NOT NULL,
                site_title TEXT,
                source_excerpt TEXT,
                roast_text TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute("ALTER TABLE roasts ADD COLUMN IF NOT EXISTS mode TEXT;")
        cur.execute("UPDATE roasts SET mode = 'roast' WHERE mode IS NULL OR mode = '';")
        cur.execute("ALTER TABLE roasts ALTER COLUMN mode SET DEFAULT 'roast';")
        cur.execute("ALTER TABLE roasts ADD COLUMN IF NOT EXISTS slug TEXT;")
        cur.execute("ALTER TABLE roasts DROP CONSTRAINT IF EXISTS roasts_normalized_url_key;")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_roasts_normalized_mode_unique "
            "ON roasts (normalized_url, mode);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_roasts_updated_at ON roasts (updated_at DESC);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_roasts_slug ON roasts (slug);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_roasts_mode ON roasts (mode);")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS domain_enrichment (
                domain TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                sources TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_domain_enrichment_updated_at "
            "ON domain_enrichment (updated_at DESC);"
        )


def normalize_mode(mode: str) -> str:
    value = (mode or "roast").strip().lower()
    return "praise" if value == "praise" else "roast"


def slugify(text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return base[:80] or "startup"


def create_unique_slug(base_text: str) -> str:
    base = slugify(base_text)
    candidate = base
    counter = 2
    with get_conn() as conn, conn.cursor() as cur:
        while True:
            cur.execute("SELECT 1 FROM roasts WHERE slug = %s LIMIT 1", (candidate,))
            if not cur.fetchone():
                return candidate
            candidate = f"{base}-{counter}"
            counter += 1


def backfill_slugs() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, normalized_url, slug FROM roasts ORDER BY id ASC")
        rows = cur.fetchall()

        used = {row[2] for row in rows if row[2]}
        updates = []
        for row in rows:
            roast_id, normalized_url, slug = row
            if slug:
                continue
            base = slugify(normalized_url)
            candidate = base
            counter = 2
            while candidate in used:
                candidate = f"{base}-{counter}"
                counter += 1
            used.add(candidate)
            updates.append((candidate, roast_id))

        if updates:
            cur.executemany("UPDATE roasts SET slug = %s WHERE id = %s", updates)


def ensure_db() -> None:
    global _db_ready
    if _db_ready:
        return
    init_db()
    backfill_slugs()
    _db_ready = True


def normalize_url(raw_url: str) -> Tuple[str, str, str]:
    text = (raw_url or "").strip()
    if not text:
        raise ValueError("URL is required.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = f"https://{text}"
    parsed = urlparse(text)
    if not parsed.netloc and parsed.path:
        text = f"https://{parsed.path}"
        parsed = urlparse(text)
    if not parsed.netloc:
        raise ValueError("Invalid URL.")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid URL protocol. Please use a website URL.")

    domain = (parsed.hostname or parsed.netloc).lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain:
        raise ValueError("Invalid URL.")

    normalized = domain
    fetch_url = f"https://{parsed.netloc}{parsed.path or ''}"
    if parsed.query:
        fetch_url = f"{fetch_url}?{parsed.query}"
    return normalized, domain, fetch_url


def _extract_text_blocks(soup: BeautifulSoup, limit: int = 35) -> str:
    text_blocks = []
    for node in soup.select("h1, h2, h3, p, li, meta[name='description']"):
        if node.name == "meta":
            piece = (node.get("content") or "").strip()
        else:
            piece = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(piece) > 30:
            text_blocks.append(piece)
        if len(text_blocks) >= limit:
            break
    return "\n".join(text_blocks)[:6000]


def fetch_search_context(domain: str) -> str:
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": f"{domain} product company startup"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        rows = []
        for result in soup.select(".result")[:5]:
            title_node = result.select_one(".result__a")
            snippet_node = result.select_one(".result__snippet")
            title = title_node.get_text(" ", strip=True) if title_node else ""
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            if title or snippet:
                rows.append(f"{title} — {snippet}".strip(" —"))
        return "\n".join(rows)[:1800]
    except requests.RequestException:
        return ""


def fetch_wiki_context(domain: str) -> str:
    keyword = domain.split(".")[0]
    if not keyword:
        return ""
    token = keyword[0].upper() + keyword[1:]
    try:
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{token}",
            timeout=8,
            headers={"User-Agent": "StartupLens/1.0"},
        )
        if response.status_code != 200:
            return ""
        payload = response.json()
        return (payload.get("extract") or "").strip()[:900]
    except requests.RequestException:
        return ""


def fetch_wiki_context_via_search(domain: str) -> str:
    keyword = domain.split(".")[0]
    if not keyword:
        return ""
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": f"{keyword} company",
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            timeout=8,
            headers={"User-Agent": "StartupLens/1.0"},
        )
        search_resp.raise_for_status()
        payload = search_resp.json()
        titles = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        if not titles:
            return ""
        title = str(titles[0]).strip()
        if not title:
            return ""
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=8,
            headers={"User-Agent": "StartupLens/1.0"},
        )
        if summary_resp.status_code != 200:
            return ""
        summary_payload = summary_resp.json()
        return (summary_payload.get("extract") or "").strip()[:900]
    except requests.RequestException:
        return ""


def fetch_wikidata_context(domain: str) -> str:
    keyword = domain.split(".")[0]
    if not keyword:
        return ""
    try:
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": keyword,
                "language": "en",
                "format": "json",
                "limit": 3,
            },
            timeout=8,
            headers={"User-Agent": "StartupLens/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("search", [])
        rows = []
        for item in results:
            label = (item.get("label") or "").strip()
            desc = (item.get("description") or "").strip()
            if label and desc:
                rows.append(f"{label}: {desc}")
        return "\n".join(rows)[:900]
    except requests.RequestException:
        return ""


def get_cached_domain_enrichment(domain: str) -> Optional[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT summary
            FROM domain_enrichment
            WHERE domain = %s
              AND updated_at > NOW() - (%s * INTERVAL '1 hour')
            """,
            (domain, DOMAIN_ENRICHMENT_CACHE_HOURS),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def set_cached_domain_enrichment(domain: str, summary: str, sources: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO domain_enrichment (domain, summary, sources, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (domain)
            DO UPDATE SET
                summary = EXCLUDED.summary,
                sources = EXCLUDED.sources,
                updated_at = NOW()
            """,
            (domain, summary, sources),
        )


def fetch_domain_enrichment(domain: str) -> str:
    cached = get_cached_domain_enrichment(domain)
    if cached:
        return cached

    wiki_summary = fetch_wiki_context(domain)
    if not wiki_summary:
        wiki_summary = fetch_wiki_context_via_search(domain)
    wikidata_summary = fetch_wikidata_context(domain)
    search_summary = fetch_search_context(domain)

    sections = []
    if wiki_summary:
        sections.append(f"[Wikipedia]\n{wiki_summary}")
    if wikidata_summary:
        sections.append(f"[Wikidata]\n{wikidata_summary}")
    if search_summary:
        sections.append(f"[Search snippets]\n{search_summary}")

    combined = "\n\n".join(sections).strip()
    if combined:
        set_cached_domain_enrichment(
            domain, combined, "wikipedia,wikidata,duckduckgo"
        )
    return combined


def fetch_site_context(url: str, domain: str) -> Tuple[str, str, str]:
    try:
        response = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
    except requests.RequestException:
        response = None

    title = ""
    excerpt = ""
    if response is not None:
        soup = BeautifulSoup(response.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
        excerpt = _extract_text_blocks(soup, limit=40)

    if len(excerpt) < 250:
        try:
            parsed = urlparse(url)
            jina_url = f"https://r.jina.ai/http://{parsed.netloc}{parsed.path or ''}"
            if parsed.query:
                jina_url = f"{jina_url}?{parsed.query}"
            jina_resp = requests.get(jina_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if jina_resp.ok:
                jina_text = re.sub(r"\s+", " ", jina_resp.text).strip()
                excerpt = f"{excerpt}\n{jina_text[:2800]}".strip()
        except requests.RequestException:
            pass

    external_context = fetch_domain_enrichment(domain)
    return title[:200], excerpt[:6000], external_context[:2200]


def generate_output(
    url: str, domain: str, title: str, excerpt: str, external_context: str, mode: str
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    mode = normalize_mode(mode)
    style_line = (
        "You are a savage startup critic. Write a very blunt roast of this startup website. "
        "Tone: ruthless, funny, insulting, but still coherent and readable."
        if mode == "roast"
        else "You are a sharp startup analyst. Write a strong praise of this startup website. "
        "Tone: specific, confident, positive, and slightly witty without being generic."
    )
    final_line = (
        "Include a final 1-line closing insult at the end."
        if mode == "roast"
        else "Include a final 1-line closing compliment at the end."
    )

    prompt = f"""
{style_line}

Target URL: {url}
Domain: {domain}
Page title: {title or "N/A"}
Extracted landing page text:
{excerpt or "No page text could be extracted."}

External internet context:
{external_context or "No external context was found."}

Rules:
- Return plain text only.
- 7-9 paragraphs.
- Keep each paragraph 2-3 sentences.
- Be specific to this startup; do not be generic.
- No bullet points, no markdown headings, no JSON.
- {final_line}
- The output must end with complete sentences and proper punctuation.
- If landing page text is sparse, rely on external context and known product context.
- Do not claim "the website has no content" unless both landing page and external context are actually empty.
""".strip()

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    response = requests.post(
        endpoint,
        timeout=60,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1600, "temperature": 0.9},
        },
    )
    response.raise_for_status()
    payload = response.json()
    text = (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return finalize_roast_text(text)


def finalize_roast_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    if cleaned[-1] in ".!?\"'":
        return cleaned

    last_stop = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
    if last_stop != -1 and last_stop > int(len(cleaned) * 0.45):
        return cleaned[: last_stop + 1].rstrip()
    return cleaned


def safe_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            return "Gemini auth failed. Check GEMINI_API_KEY."
        if status == 404:
            return "Gemini model not found. Check GEMINI_MODEL."
        if status == 429:
            return (
                "Sorry, Gemini free-tier rate limit (429) is hit right now. "
                "Please try again in a bit."
            )
        if status and status >= 500:
            return "Gemini service is temporarily unavailable."
        return "Gemini request failed."
    if isinstance(exc, requests.RequestException):
        return "Network error while contacting the URL or Gemini."
    return str(exc)

def get_roast_by_normalized(normalized_url: str, mode: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, slug, normalized_url, original_url, display_domain, site_title, source_excerpt,
                   roast_text, request_count, created_at, updated_at, mode
            FROM roasts
            WHERE normalized_url = %s AND mode = %s
            """,
            (normalized_url, normalize_mode(mode)),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "slug": row[1],
            "normalized_url": row[2],
            "original_url": row[3],
            "display_domain": row[4],
            "site_title": row[5] or "",
            "source_excerpt": row[6] or "",
            "roast_text": row[7],
            "request_count": row[8],
            "created_at": row[9],
            "updated_at": row[10],
            "mode": row[11],
        }


def touch_roast(roast_id: int) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE roasts
            SET request_count = request_count + 1,
                updated_at = NOW()
            WHERE id = %s
            """,
            (roast_id,),
        )


def set_roast_slug(roast_id: int, slug: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE roasts SET slug = %s WHERE id = %s", (slug, roast_id))


def insert_roast(
    mode: str,
    slug: str,
    normalized_url: str,
    original_url: str,
    display_domain: str,
    site_title: str,
    source_excerpt: str,
    roast_text: str,
) -> str:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO roasts (
                mode, slug, normalized_url, original_url, display_domain, site_title, source_excerpt, roast_text
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING slug
            """,
            (
                normalize_mode(mode),
                slug,
                normalized_url,
                original_url,
                display_domain,
                site_title,
                source_excerpt,
                roast_text,
            ),
        )
        return cur.fetchone()[0]


def get_recent_roasts(limit: int = 12):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, mode, display_domain, roast_text, request_count, updated_at
            FROM roasts
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "slug": row[0],
                "mode": row[1],
                "display_domain": row[2],
                "teaser": (row[3].splitlines()[0][:120] if row[3] else "").strip(),
                "request_count": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]


def get_total_destroyed() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(request_count), 0) FROM roasts")
        return int(cur.fetchone()[0])


def process_destroy(raw_url: str, mode: str) -> Tuple[Optional[str], Optional[str], int]:
    mode = normalize_mode(mode)
    try:
        normalized_url, display_domain, fetch_url = normalize_url(raw_url)
    except ValueError as exc:
        return None, str(exc), 400

    existing = get_roast_by_normalized(normalized_url, mode)
    if existing:
        if not existing["slug"]:
            fallback_slug = create_unique_slug(existing["normalized_url"])
            set_roast_slug(existing["id"], fallback_slug)
            existing["slug"] = fallback_slug
        touch_roast(existing["id"])
        return str(existing["slug"]), None, 200

    try:
        site_title, source_excerpt, external_context = fetch_site_context(fetch_url, display_domain)
        roast_text = generate_output(
            fetch_url, display_domain, site_title, source_excerpt, external_context, mode
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        return None, safe_error_message(exc), status
    except Exception as exc:
        return None, safe_error_message(exc), 502

    unique_slug = create_unique_slug(display_domain)
    roast_slug = insert_roast(
        mode=mode,
        slug=unique_slug,
        normalized_url=normalized_url,
        original_url=fetch_url,
        display_domain=display_domain,
        site_title=site_title,
        source_excerpt=source_excerpt,
        roast_text=roast_text,
    )
    return roast_slug, None, 200


@app.route("/", methods=["GET"])
def home():
    ensure_db()
    error = request.args.get("error", "")
    recent = get_recent_roasts()
    total_destroyed = get_total_destroyed()
    return render_template(
        "index.html",
        recent=recent,
        total_destroyed=total_destroyed,
        error=error,
    )


@app.route("/destroy", methods=["POST"])
def destroy():
    ensure_db()
    raw_url = request.form.get("url", "")
    mode = normalize_mode(request.form.get("mode", "roast"))
    roast_slug, error, _status_code = process_destroy(raw_url, mode)
    if error:
        return redirect(url_for("home", error=error))
    return redirect(url_for("result_detail", mode=mode, slug=roast_slug))


@app.route("/api/destroy", methods=["POST"])
def destroy_api():
    ensure_db()
    payload = request.get_json(silent=True) or {}
    mode = normalize_mode(payload.get("mode", "roast"))
    roast_slug, error, status_code = process_destroy(payload.get("url", ""), mode)
    if error:
        return jsonify({"ok": False, "error": error}), status_code
    return jsonify(
        {"ok": True, "redirect_url": url_for("result_detail", mode=mode, slug=roast_slug)}
    )


@app.route("/roast/<slug>", methods=["GET"])
def roast_detail_legacy(slug: str):
    return redirect(url_for("result_detail", mode="roast", slug=slug))


@app.route("/<mode>/<slug>", methods=["GET"])
def result_detail(mode: str, slug: str):
    mode = normalize_mode(mode)
    ensure_db()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, slug, mode, display_domain, roast_text, original_url, site_title, created_at
            FROM roasts
            WHERE slug = %s AND mode = %s
            """,
            (slug, mode),
        )
        row = cur.fetchone()
        if not row and slug.isdigit():
            cur.execute(
                """
                SELECT id, slug, mode, display_domain, roast_text, original_url, site_title, created_at
                FROM roasts
                WHERE id = %s
                """,
                (int(slug),),
            )
            legacy_row = cur.fetchone()
            if legacy_row:
                return redirect(
                    url_for("result_detail", mode=legacy_row[2], slug=legacy_row[1])
                )
            row = None

    if not row:
        abort(404)

    roast = {
        "id": row[0],
        "slug": row[1],
        "mode": row[2],
        "display_domain": row[3],
        "roast_text": row[4],
        "original_url": row[5],
        "site_title": row[6] or "",
        "created_at": row[7],
    }
    paragraphs = [p.strip() for p in roast["roast_text"].split("\n\n") if p.strip()]
    return render_template(
        "roast.html", roast=roast, paragraphs=paragraphs, result_mode=mode
    )


if __name__ == "__main__":
    ensure_db()
    app.run(host="0.0.0.0", port=5001, debug=True)
