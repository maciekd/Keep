"""A DuckDuckGo web search without an API key, the report LLM's fact-checking tool.

Scrapes the no-JS endpoint html.duckduckgo.com/html/. A cold request without cookies meets the
bot wall, so the session first visits duckduckgo.com/ for its cookies (once per process, again
after a blocked answer). A failed search answers an empty list: the model is told there were no
results and carries on, marking what it could not verify.
"""
import html
import http.cookiejar
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from common.logging import info, trace

DDG_URL = "https://html.duckduckgo.com/html/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
TITLE_RE = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")

_lock = threading.Lock()
_opener = None


def _primed_opener():
    global _opener
    with _lock:
        if _opener is None:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            opener.open(urllib.request.Request("https://duckduckgo.com/", headers={"User-Agent": UA}), timeout=10).read()
            _opener = opener
        return _opener


def _forget_session():
    global _opener
    with _lock:
        _opener = None


def _clean(text: str) -> str:
    return html.unescape(TAG_RE.sub("", text)).strip()


def _resolve(href: str) -> str:
    """DDG links are //duckduckgo.com/l/?uddg=<the url>&rut=…"""
    if href.startswith("//"):
        href = "https:" + href
    return urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]


def parse_results(body: str, max_results: int) -> list[dict]:
    snippets = SNIPPET_RE.findall(body)
    results = []
    for i, (href, title) in enumerate(TITLE_RE.findall(body)[:max_results]):
        if _clean(title):
            results.append({"title": _clean(title), "url": _resolve(href),
                            "snippet": _clean(snippets[i]) if i < len(snippets) else ""})
    return results


def search(query: str, max_results: int = 4, timeout: float = 10, retries: int = 2) -> list[dict]:
    """[{title, url, snippet}] for `query`; [] when DuckDuckGo blocks or cannot be reached."""
    url = f"{DDG_URL}?{urllib.parse.urlencode({'q': query})}"
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://duckduckgo.com/",
                                                           "Accept-Language": "en-US,en;q=0.9"})
            body = _primed_opener().open(request, timeout=timeout).read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            trace(f"Podcasts: DuckDuckGo search {query!r} failed (attempt {attempt + 1}): {e}")
            time.sleep(1.5 * (attempt + 1))
            continue
        if "anomaly" in body.lower() and "result__a" not in body:
            trace(f"Podcasts: DuckDuckGo bot wall for {query!r}, new session (attempt {attempt + 1})")
            _forget_session()
            time.sleep(1.5 * (attempt + 1))
            continue
        results = parse_results(body, max_results)
        trace(f"Podcasts: web_search({query!r}) -> {len(results)} result(s)")
        return results
    info(f"Podcasts: web_search({query!r}) gave up after {retries + 1} attempts — no results")
    return []


def format_results(query: str, results: list[dict]) -> str:
    if not results:
        return (f'No results found for query: "{query}" (search may be rate-limited — proceed with available '
                f'knowledge and flag any resulting uncertainty explicitly).')
    lines = [f'Search results for "{query}":']
    lines += [f'{i}. {r["title"]} — {r["url"]}\n   {r["snippet"]}' for i, r in enumerate(results, 1)]
    return "\n".join(lines)
