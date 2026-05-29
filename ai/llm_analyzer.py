"""ai/llm_analyzer.py

Claude Haiku 4.5가 뉴스 기사를 읽고 위험 정보를 '구조화된 JSON'으로 추출합니다.

핵심 설계:
  - 모델: claude-haiku-4-5 (가장 빠르고 저렴 — $1/$5 per 1M tokens; 30 기사 ≈ $0.05)
  - 출력: Pydantic 모델로 스키마 강제 -> 잘못된 JSON 절대 안 옴 (client.messages.parse())
  - 캐시: 같은 기사를 재분석 안 함 (data/llm_cache.json, URL 기반 키)
  - 프롬프트 캐시: system 블록에 cache_control 부착 -> 미래의 긴 프롬프트에 대비.
    (현재 프롬프트가 짧아서 Haiku의 4096 토큰 최소치 미달 -> 지금은 no-op,
     프롬프트가 늘어나면 자동으로 활성화됨.)
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

CACHE_PATH = Path("data/llm_cache.json")
MODEL = "claude-haiku-4-5"


# ---- 응답 스키마 (Pydantic 모델) ---------------------------------------
# Anthropic SDK가 이 스키마를 JSON Schema로 자동 변환해서 모델에 전달.
# 응답은 이 모델로 검증되므로 잘못된 JSON 형식이 절대 들어오지 않습니다.

class Hazard(BaseModel):
    """뉴스에서 뽑아낸 위험 한 건."""
    type: Literal["shelling", "enemy_forces", "destroyed_building",
                  "enemy_drone", "landmine"]
    location_description: str = Field(description="구체적 위치 (예: 'central Bakhmut')")
    lat: float = Field(description="대략적 위도. 모르면 Bakhmut 중심 48.59")
    lon: float = Field(description="대략적 경도. 모르면 Bakhmut 중심 38.00")
    severity: int = Field(description="1~10, 10이 가장 위험")
    time: str = Field(default="", description="대략적 시각/날짜")


class NewsExtraction(BaseModel):
    """뉴스 기사 한 건의 분석 결과."""
    hazards: list[Hazard] = Field(default_factory=list)
    road_closures: list[str] = Field(default_factory=list)
    civilian_advisory: str = ""


# ---- Anthropic 클라이언트 (지연 초기화) ---------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        # ANTHROPIC_API_KEY 우선; 없으면 OPENAI_API_KEY 슬롯에 들어 있는 키로 폴백
        # (지승님이 .env 변수명을 ANTHROPIC_API_KEY 로 바꾸면 더 깔끔하지만, 지금도 동작)
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY (또는 OPENAI_API_KEY)가 .env에 없습니다")
        from anthropic import Anthropic
        _client = Anthropic(api_key=key)
    return _client


# ---- 캐시 ---------------------------------------------------------------

def _load_cache():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def _article_key(article):
    """기사별 캐시 키: URL이 있으면 URL, 없으면 제목 해시."""
    url = article.get("url", "")
    if url and not url.startswith("mock://"):
        return url
    return "h:" + hashlib.sha1(
        (article.get("title", "") + article.get("description", "")).encode("utf-8")
    ).hexdigest()


# ---- 메인 API -----------------------------------------------------------

def extract_hazards_from_article(article, hazard_module):
    """기사 한 건 -> 구조화된 위험 정보 (캐시됨)."""
    cache = _load_cache()
    key = _article_key(article)
    if key in cache:
        return cache[key]

    user_text = (
        f"Article title: {article.get('title','')}\n"
        f"Article description: {article.get('description','')}\n"
        f"Article content: {article.get('content','')}\n"
        f"Source: {article.get('source','')}, Published: {article.get('published_at','')}"
    )

    try:
        client = _get_client()
        # messages.parse() = messages.create() + 응답 자동 검증.
        # output_format에 Pydantic 모델을 주면 그 스키마로 JSON 강제.
        resp = client.messages.parse(
            model=MODEL,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": hazard_module.llm_prompt,
                # 시스템 프롬프트 캐시 마커 (현재 길이가 짧아 활성화 안 되지만 미래 대비)
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_text}],
            output_format=NewsExtraction,
        )
        parsed = resp.parsed_output.model_dump()
    except Exception as e:
        print(f"[LLM] '{article.get('title','?')[:50]}' 분석 실패: {e}")
        parsed = {"hazards": [], "road_closures": [], "civilian_advisory": ""}

    cache[key] = parsed
    _save_cache(cache)
    return parsed


def extract_hazards_from_all(articles, hazard_module):
    """여러 기사 -> 통합된 (위험 목록, 도로 폐쇄 목록)."""
    hazards = []
    closures = []
    for art in articles:
        result = extract_hazards_from_article(art, hazard_module)
        for h in result.get("hazards", []) or []:
            h["_source"] = art.get("source", "")
            h["_published_at"] = art.get("published_at", "")
            h["_article_url"] = art.get("url", "")
            hazards.append(h)
        for road in result.get("road_closures", []) or []:
            if road:
                closures.append({"road": road,
                                 "source": art.get("source", ""),
                                 "url": art.get("url", "")})
    return hazards, closures
