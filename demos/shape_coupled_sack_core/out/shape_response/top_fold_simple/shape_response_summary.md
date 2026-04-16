# Shape Response Demo Summary

- scenario: `top_fold_simple`
- target_site: `grasp_fold_1`
- video: frame sequence only

## 핵심 변화
- upper_half_width delta: `0.00495 m`
- lower_half_width delta: `0.00742 m`
- local shoulder angle delta: `-2.57 deg`
- bottom_sag delta: `0.00013 m`
- payload_y delta: `-0.01658 m`
- bag_com_z delta: `-0.01525 m`

## 해석
이 데모는 full soft cloth가 아니라, 접촉과 지지가 들어오면 패널 힌지, 하부 sling, 내부 payload가 함께 반응하는 reduced-order shape coupling입니다.
따라서 pure material simulator라고 주장하지 않고, support-state formation 평가에 필요한 형상 변화와 하중 재분배를 안정적으로 보여주는 task-driven surrogate로 사용합니다.