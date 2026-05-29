"""ai/news_collector.py

NewsAPI에서 Bakhmut 관련 실시간 영문 뉴스를 받아옵니다.

무료 NewsAPI 한계:
  - 하루 100요청 (우리는 키워드 몇 개만 쓰니 충분)
  - 기사는 약 24시간 지연
  - 최근 1개월 이내 기사만 검색 가능
  - localhost 전용 (우리 로컬 데모에는 문제없음)

결과는 data/news_cache.json 에 캐시:
  - 같은 결과로 재현 가능 -> KCF 데모 신뢰성
  - 무료 할당량 절약
  - --refresh-news 플래그로 강제 갱신
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
CACHE_PATH = Path("data/news_cache.json")


def fetch_news(keywords, days=14, page_size=15, language="en", force_refresh=False):
    """뉴스 기사 목록을 반환. 캐시가 있고 force_refresh=False면 캐시 사용."""
    if CACHE_PATH.exists() and not force_refresh:
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"[NewsAPI] 캐시 사용: {len(cached)}개 기사 (--refresh-news 로 갱신)")
        return cached

    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("[NewsAPI] NEWSAPI_KEY 없음 -> mock 뉴스 사용")
        return _mock_news()

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    all_articles = []
    seen_urls = set()

    for kw in keywords:
        params = {
            "q": kw,
            "from": from_date,
            "language": language,
            "sortBy": "relevancy",
            "pageSize": page_size,
            "apiKey": api_key,
        }
        try:
            r = requests.get(NEWSAPI_URL, params=params, timeout=20)
            if r.status_code != 200:
                print(f"[NewsAPI] '{kw}' 응답 {r.status_code}: {r.text[:120]}")
                continue
            for art in r.json().get("articles", []):
                url = art.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                all_articles.append({
                    "title": art.get("title", "") or "",
                    "description": art.get("description", "") or "",
                    "content": art.get("content", "") or "",
                    "published_at": art.get("publishedAt", "") or "",
                    "source": (art.get("source") or {}).get("name", "") or "",
                    "url": url,
                    "query": kw,
                })
        except Exception as e:
            print(f"[NewsAPI] '{kw}' 요청 실패: {e}")

    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(all_articles, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[NewsAPI] {len(all_articles)}개 기사 수집 -> {CACHE_PATH}")
    return all_articles


def _mock_news():
    """API 없이도 데모 가능하게 — 합성 뉴스 몇 건."""
    return [
        {
            "title": "Russian artillery strikes central Bakhmut, residential block hit",
            "description": "Heavy shelling reported in the center of Bakhmut. Multiple residential buildings damaged.",
            "content": "",
            "published_at": "2025-12-01T10:00:00Z",
            "source": "Mock-Reuters",
            "url": "mock://1",
            "query": "Bakhmut",
        },
        {
            "title": "Drone strike reported near Bakhmut highway",
            "description": "Ukrainian sources report a Russian drone strike near the M-03 highway approaching Bakhmut from the west.",
            "content": "",
            "published_at": "2025-12-02T08:00:00Z",
            "source": "Mock-AP",
            "url": "mock://2",
            "query": "Bakhmut",
        },
    ]
