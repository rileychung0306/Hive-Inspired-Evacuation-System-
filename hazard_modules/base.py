"""hazard_modules/base.py

재난(위험) 종류를 정의하는 '교체 가능한 블록'의 뼈대입니다.

이 파일은 "어떤 재난을 다룰지"에 대한 공통 형식(틀)만 정합니다.
분쟁(conflict), 산불(wildfire) 같은 구체적인 재난은 이 형식을 그대로 따릅니다.
그래서 나머지 시스템(드론, 셀룰러 오토마타, 라우팅, 시민 앱)은 이 형식만 알면
재난 종류가 바뀌어도 코드를 거의 고치지 않아도 됩니다.
이것을 'hazard-agnostic(재난 종류와 무관한) 설계'라고 부릅니다.  -> 우리 프로젝트의 차별점 4축 중 하나.
"""

from dataclasses import dataclass


@dataclass
class HazardSpec:
    """위험 '한 종류'의 성질을 담는 상자.

    예) 포격(shelling)은 빠르게 번지고 금방 사그라들지만,
        지뢰(landmine)는 거의 움직이지 않고 사라지지도 않습니다.
        이런 차이를 아래 숫자들로 표현합니다.
    """

    name: str                 # 코드에서 쓰는 이름 (예: "shelling")
    display_name: str         # 화면/발표에서 보여줄 한글 이름 (예: "포격")
    color: tuple              # 지도에 칠할 색 (R, G, B)
    spread_prob: float = 0.0  # 매 스텝마다 이웃 칸으로 번질 확률 (0~1)
    neighborhood: int = 8     # 이웃을 몇 칸으로 볼지 (4 = 상하좌우, 8 = 대각선 포함)
    decay_per_step: float = 0.0   # 매 스텝마다 위험도가 줄어드는 비율 (0~1)
    mobile: bool = False      # True면 위치가 이동함 (예: 적 드론)
    static: bool = False      # True면 거의 변하지 않음 (예: 지뢰, 파괴된 건물)
    spawn_rate: float = 0.0   # 시간이 지나며 새로 생겨날 확률 (예: 새 지뢰)


class HazardModule:
    """재난 '한 종류 전체'를 정의하는 블록의 부모(공통) 클래스.

    자식 클래스(예: ConflictModule)는 아래 4가지를 채워 넣습니다.
      - name           : 모듈 이름 (예: "conflict")
      - hazards        : HazardSpec 목록 (이 재난이 가진 위험 종류들)
      - news_keywords  : 뉴스 검색에 쓸 키워드 (Phase 3에서 사용)
      - llm_prompt     : LLM(GPT)에게 줄 지시문 (Phase 3에서 사용)
    """

    name: str = "base"
    hazards: list = []
    news_keywords: list = []
    llm_prompt: str = ""

    @property
    def hazard_names(self) -> list:
        """위험 종류의 '코드 이름' 목록. 예: ["enemy_forces", "shelling", ...]"""
        return [h.name for h in self.hazards]

    @property
    def num_hazards(self) -> int:
        """이 재난이 다루는 위험 종류의 개수."""
        return len(self.hazards)

    def get(self, name: str) -> HazardSpec:
        """이름으로 위험 한 종류(HazardSpec)를 찾아 돌려줍니다."""
        for h in self.hazards:
            if h.name == name:
                return h
        raise KeyError(f"알 수 없는 위험 종류입니다: {name}")
