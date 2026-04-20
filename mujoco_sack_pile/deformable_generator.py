from __future__ import annotations

"""호환성용 별칭 모듈.

이 파일은 예전 이름을 유지하기 위한 thin wrapper다.
정량 benchmark 본체는 아니며, 실제 재질 시뮬레이터로 해석하면 안 된다.
"""

from .qualitative_reference import QualitativeReferenceGenerator, ReferenceShellSpec


DeformableSackSpec = ReferenceShellSpec


class DeformablePileGenerator(QualitativeReferenceGenerator):
    """이전 API 이름과의 호환을 위한 별칭 클래스."""

