# 🐝 Hive-Inspired Evacuation System

꿀벌의 집단 의사결정(quorum sensing) 방식으로 움직이는 **드론 무리**가 분쟁지역을 정찰하고,
동시에 **AI(LLM)** 가 실시간 영문 뉴스를 읽어, 시민에게 **가장 안전한 대피 경로**를 안내하는 시스템.

> 한국 코드페어(KCF) 출품작 · 제출 마감 2026-06-20

## 차별화 4축
1. **알고리즘** — PSO 대신 꿀벌 민주주의(Seeley quorum sensing)
2. **데이터 융합** — 드론 시각 정보 + LLM 뉴스 분석
3. **다중 위험** — 5가지 분쟁 위험을 동시에 + 시간에 따른 확산(셀룰러 오토마타)
4. **시민 중심** — 오프라인에서도 쓰는 모바일 대피 안내 앱

## 폴더 구조
```
hazard_modules/   교체 가능한 재난 정의 (현재: 분쟁)
simulation/       도시 + 위험 확산 + 드론          (Phase 1-2)
ai/               뉴스 수집 + LLM 분석 + Bee voting (Phase 3)
routing/          OSM 도로망 + 위험 가중 A*         (Phase 4)
app/              Streamlit 시민 앱 + QR 인쇄        (Phase 5)
experiments/      Bee vs PSO 비교 실험              (Phase 6)
data/             ACLED/지도 등 데이터
config/           공통 설정값
main.py           전체 통합 실행 입구
```

## 설치 & 실행
```bash
# 1) 가상환경 켜기 (이미 만들어져 있음)
source kcf-env/bin/activate

# 2) (처음 한 번) 키 파일 만들기
cp .env.example .env   # 그리고 .env 안에 실제 키 입력

# 3) 실행
python main.py
```

## 개발 단계 (로드맵)
- [x] **Phase 0** 환경 설정 + 재난 모듈 뼈대
- [ ] **Phase 1** 도시 격자 + 5가지 위험 확산
- [ ] **Phase 2** 드론 군집 + 꿀벌 투표
- [ ] **Phase 3** 뉴스 + LLM 융합
- [ ] **Phase 4** 위험 가중 A* 라우팅
- [ ] **Phase 5** 시민 앱
- [ ] **Phase 6** 통합 + 비교 실험
- [ ] **Phase 7** 제출 서류
