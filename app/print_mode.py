"""app/print_mode.py

오프라인 백업용 안전 경로 PDF 생성기.

내용 (A5 한 장):
  - 제목 + 사용자 분류
  - 출발지/도착지 좌표
  - 거리, 도보 시간, 평균 위험도
  - 시스템 권고 (도달 불가 시 차량 지원 안내)
  - QR 코드: Google Maps 에 대피소 좌표 열기 (스마트폰으로 스캔)

통신 두절·노인·저시력 사용자에게도 종이 한 장으로 안내 가능.
"""

import io
from datetime import datetime

import qrcode
from reportlab.lib.pagesizes import A5
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# 한글 폰트 등록 (없으면 영어 폴백)
_KOREAN_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_KOREAN_FONT = None
for _p in _KOREAN_FONT_PATHS:
    try:
        pdfmetrics.registerFont(TTFont("AppleGothic", _p))
        _KOREAN_FONT = "AppleGothic"
        break
    except Exception:
        continue


def _font(bold: bool = False) -> str:
    """한글 가능하면 한글 폰트, 아니면 Helvetica."""
    if _KOREAN_FONT:
        return _KOREAN_FONT
    return "Helvetica-Bold" if bold else "Helvetica"


def generate_route_pdf(route: dict, start_lat: float, start_lon: float) -> bytes:
    """안전 경로 PDF 생성. route 는 risk_astar.route_for_civilian 결과."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A5)
    W, H = A5
    margin = 1.5 * cm
    has_kr = _KOREAN_FONT is not None

    # 헤더
    c.setFillColorRGB(0.95, 0.55, 0.1)
    c.rect(0, H - 2.2 * cm, W, 2.2 * cm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(_font(bold=True), 18)
    c.drawString(margin, H - 1.4 * cm,
                 "🐝 Hive Evac — 안전 대피 경로" if has_kr
                 else "Hive Evac — Emergency Route")

    c.setFillColorRGB(0, 0, 0)
    y = H - 3.2 * cm

    def line(label_kr, label_en, value, big=False):
        nonlocal y
        c.setFont(_font(), 14 if big else 11)
        text = f"{label_kr if has_kr else label_en}: {value}"
        c.drawString(margin, y, text)
        y -= 0.8 * cm if big else 0.6 * cm

    # 사용자 분류
    profile_lbl = route.get("profile_label", route.get("profile_key", "?"))
    line("분류", "Profile", profile_lbl, big=True)

    # 출발 / 도착
    line("출발 좌표", "Start", f"{start_lat:.4f}, {start_lon:.4f}")

    if route.get("shelter") and route.get("path_coords"):
        sh = route["shelter"]
        line("대피소", "Destination", sh["name"], big=True)
        line("대피소 좌표", "Coords", f"{sh['lat']:.4f}, {sh['lon']:.4f}")
        line("거리", "Distance", f"{route['total_meters']:.0f} m")
        line("예상 소요 시간", "Travel time", f"{route['travel_time_min']:.1f} 분"
             if has_kr else f"{route['travel_time_min']:.1f} min")
        line("평균 위험도", "Avg risk", f"{route['avg_risk']:.2f}")
        qr_target = f"https://maps.google.com/?q={sh['lat']},{sh['lon']}"
    else:
        # 도달 불가
        c.setFillColorRGB(0.7, 0.1, 0.1)
        c.setFont(_font(bold=True), 14)
        msg = ("⚠ 도달 가능한 대피소 없음" if has_kr
               else "WARNING: No reachable shelter")
        c.drawString(margin, y, msg)
        y -= 0.8 * cm
        c.setFont(_font(), 11)
        msg2 = ("시스템 권고: 차량 지원 / 이웃 도움 필요"
                if has_kr else "System advice: vehicle / neighbor assistance needed")
        c.drawString(margin, y, msg2)
        y -= 0.6 * cm
        c.setFillColorRGB(0, 0, 0)
        qr_target = f"https://maps.google.com/?q={start_lat},{start_lon}"

    # 안내 텍스트
    y -= 0.4 * cm
    c.setFont(_font(), 9)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.drawString(margin, y,
                 "QR 코드를 스마트폰으로 스캔하면 지도가 열립니다."
                 if has_kr else "Scan the QR code with a phone to open the map.")

    # QR 코드 (우하단)
    img = qrcode.make(qr_target)
    qr_buf = io.BytesIO()
    img.save(qr_buf, format="PNG")
    qr_buf.seek(0)
    qr_size = 4.5 * cm
    c.drawImage(ImageReader(qr_buf),
                W - margin - qr_size, margin, qr_size, qr_size)
    c.setFont(_font(), 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawRightString(W - margin, margin - 0.3 * cm,
                      datetime.now().strftime("생성: %Y-%m-%d %H:%M"
                      if has_kr else "Generated: %Y-%m-%d %H:%M"))

    # 푸터
    c.setFont(_font(), 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(margin, margin,
                 "Hive Evac System — Bakhmut 안전 대피 시스템 (KCF 2026)"
                 if has_kr else "Hive Evac System — Bakhmut Evacuation (KCF 2026)")

    c.save()
    buf.seek(0)
    return buf.getvalue()
