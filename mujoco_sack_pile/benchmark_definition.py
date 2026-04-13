from __future__ import annotations

from dataclasses import asdict, dataclass


RESEARCH_QUESTION = "support-state formation under shape and pile uncertainty"
BENCHMARK_NAME = "Sack Support-State Benchmark"
BENCHMARK_SCOPE = "task-driven benchmark"
PILE_DIFFICULTIES = (
    "top_exposed",
    "side_exposed",
    "partially_buried",
    "leaning_wedged",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """타깃 자루가 속한 benchmark case 라벨을 담는다."""

    benchmark_name: str
    research_question: str
    shape_family: str
    pile_difficulty: str
    case_id: str
    tags: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


def infer_pile_difficulty(exposed_face: str, stack_level: int, tilt_mag: float) -> str:
    """초기 노출과 기울기, 적재 상태로 pile difficulty를 추정한다."""

    if stack_level > 0 and tilt_mag >= 0.35:
        return "leaning_wedged"
    if exposed_face == "top":
        return "top_exposed"
    if exposed_face == "side":
        return "side_exposed"
    return "partially_buried"


def validate_pile_difficulty(pile_difficulty: str) -> str:
    """지원하는 pile difficulty 문자열만 허용한다."""

    if pile_difficulty not in PILE_DIFFICULTIES:
        choices = ", ".join(PILE_DIFFICULTIES)
        raise ValueError(f"지원하지 않는 pile difficulty입니다: {pile_difficulty}. 사용 가능 값: {choices}")
    return pile_difficulty


def build_case_id(shape_family: str, pile_difficulty: str) -> str:
    """shape family와 pile difficulty를 합쳐 case id를 만든다."""

    return f"{shape_family}:{pile_difficulty}"


def build_benchmark_case(
    *,
    shape_family: str,
    pile_difficulty: str,
    top_collapse: float,
    side_bulge: float,
    tilt_mag: float,
) -> BenchmarkCase:
    """형상 family와 pile difficulty를 benchmark case로 정리한다."""

    pile_difficulty = validate_pile_difficulty(pile_difficulty)
    tags: list[str] = [shape_family, pile_difficulty]
    if top_collapse >= 0.015:
        tags.append("top_collapse")
    if side_bulge >= 0.015:
        tags.append("side_asymmetry")
    if pile_difficulty == "leaning_wedged":
        tags.append("stacked_contact")
    if pile_difficulty == "partially_buried":
        tags.append("cover_contact")
    if tilt_mag >= 0.35 and pile_difficulty != "leaning_wedged":
        tags.append("large_initial_tilt")

    return BenchmarkCase(
        benchmark_name=BENCHMARK_NAME,
        research_question=RESEARCH_QUESTION,
        shape_family=shape_family,
        pile_difficulty=pile_difficulty,
        case_id=build_case_id(shape_family, pile_difficulty),
        tags=tuple(tags),
    )
