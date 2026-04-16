# Shape Response Demo Summary

- scenario: `post_separation_sag`
- target_site: `grasp_shoulder_02`
- video: frame sequence only

## 핵심 변화
- upper_half_width delta: `-0.03442 m`
- lower_half_width delta: `-0.05216 m`
- local shoulder angle delta: `-3.77 deg`
- bottom_sag delta: `-0.00570 m`
- payload_y delta: `-0.01493 m`
- bag_com_z delta: `0.03946 m`

## 해석
이 데모는 full soft cloth가 아니라, 접촉과 지지가 들어오면 패널 힌지, 하부 sling, 내부 payload가 함께 반응하는 reduced-order shape coupling입니다.
따라서 pure material simulator라고 주장하지 않고, support-state formation 평가에 필요한 형상 변화와 하중 재분배를 안정적으로 보여주는 task-driven surrogate로 사용합니다.