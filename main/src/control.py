#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Controller:
    """
    control.py
    - ROS 노드 아님. main.py에서 import 해서 사용.
    - 입력: offset, mode
    - 출력: steer, speed

    설계 포인트:
    - 모드별로 함수 분리(나중에 알고리즘 완전히 교체하기 쉬움)
    - speed는 모드별 고정
    - steer는 (neutral + P*offset) 형태를 기본으로 두되
      모드별로 P/제한값을 다르게 설정 가능
    """

    def __init__(self):
        # ====== 조향 기본값/제한 ======
        # 너가 말한 제한을 기본값으로 넣어둠
        self.steer_neutral = 90
        self.steer_min = 45
        self.steer_max = 135

        # ====== offset -> steer 변환 스케일(모드별 P 게인) ======
        # offset 단위가 "픽셀"이면 P를 작게, "정규화(-1~1)"면 P를 크게 해야 함
        # 지금 offset은 (target_x - cx)라서 픽셀 단위일 확률 높음
        self.kp_mode0 = 0.6
        self.kp_mode1 = 0.20
        self.kp_mode2 = 0.11
        self.kp_mode4 = 0.25
        self.kp_mode5 = 0.18

        # ====== 모드별 속도(고정) ======
        # speed 단위는 너희 모터 노드 규격에 맞춰서 튜닝 필요
        self.speed_mode0 = 99
        self.speed_mode1 = 90
        self.speed_mode2 = 100
        self.speed_mode4 = 90
        self.speed_mode5 = 90

        # ====== 모드별 steer 제한(더 강하게 제한하고 싶으면 여기서 조절 가능) ======
        # 예: 신호대기 후 출발/AR 주행 등에서 급조향 제한
        self.steer_limit_mode0 = (45, 135)
        self.steer_limit_mode1 = (45, 135)
        self.steer_limit_mode2 = (50, 130)
        self.steer_limit_mode4 = (45, 135)
        self.steer_limit_mode5 = (55, 125)

    # ---------------------------
    # public API
    # ---------------------------
    def compute(self, offset: float, mode: int):
        """
        메인에서 호출하는 함수.
        return: steer(int), speed(int)
        """
        if mode == 0:
            return self._control_mode0(offset)
        elif mode == 1:
            return self._control_mode1(offset)
        elif mode == 2:
            return self._control_mode2(offset)
        elif mode == 4:
            return self._control_mode4(offset)
        elif mode == 5:
            return self._control_mode5(offset)
        else:
            # 모르는 모드면 안전하게 중립 + 저속
            steer = self.steer_neutral
            speed = 95
            return int(steer), int(speed)

    def _offset_to_steer_nonlinear(
        self,
        offset: float,
        max_offset: float = 150.0,
        center_steer: float = 90.0,
        steer_range: float = 45.0,
        gamma: float = 2.0
    ):
        """
        offset (-max_offset ~ +max_offset)을
        steer (45 ~ 135)로 비선형 변환
        """

        # 1. offset 제한
        offset = np.clip(offset, -max_offset, max_offset)

        # 2. 정규화 (-1 ~ 1)
        norm = offset / max_offset

        # 3. 비선형 변환 (중앙 둔감, 끝 민감)
        curved = np.sign(norm) * (abs(norm) ** gamma)

        # 4. steer 계산
        steer = center_steer + curved * steer_range

        # 5. 최종 안전 클램프
        steer = np.clip(steer, 45, 135)

        return float(steer)
    
    # ---------------------------
    # mode별 함수(나중에 알고리즘 교체 쉬움)
    # ---------------------------
    def _control_mode0(self, offset: float):
        steer = self._offset_to_steer_nonlinear(
            offset=offset,
            max_offset=150.0,
            gamma=2.0          # 둔하면 낮추고 예민하면 높이면 됨
        )
        speed = self.speed_mode0
        return steer, speed


    def _control_mode1(self, offset: float):
        # mode1: (예) 노란 정지선/갈림길 좌우 등 mode0과 비슷하지만 kp 다르게
        steer = self._p_control(offset, self.kp_mode1, self.steer_limit_mode1)
        speed = self.speed_mode1
        return steer, speed

    def _control_mode2(self, offset: float):
        # mode2: (예) 흰 차선 주행(완전 다른 알고리즘으로 바꿀 예정이라 함수 분리)
        steer = self._p_control(offset, self.kp_mode2, self.steer_limit_mode2)
        speed = self.speed_mode2
        return steer, speed

    def _control_mode4(self, offset: float):
        # mode4: 라바콘 주행(지금은 P만 다르게)
        steer = self._p_control(offset, self.kp_mode4, self.steer_limit_mode4)
        speed = self.speed_mode4
        return steer, speed

    def _control_mode5(self, offset: float):
        # mode5: AR 주행(지금은 P만 다르게 + steer 제한 더 좁게)
        steer = self._p_control(offset, self.kp_mode5, self.steer_limit_mode5)
        speed = self.speed_mode5
        return steer, speed

    # ---------------------------
    # 공통 유틸
    # ---------------------------
    def _p_control(self, offset: float, kp: float, steer_limit: tuple):
        """
        가장 기본 P 제어:
          steer = neutral + kp * offset

        offset이 픽셀 단위면:
          kp를 작게(0.05~0.3 등)
        offset이 -1~1 정규화면:
          kp를 크게(10~40 등)
        """
        raw = self.steer_neutral + kp * offset

        mn, mx = steer_limit
        raw = self._clamp(raw, mn, mx)

        # 최종적으로 전체 안전 범위도 한 번 더 클램프
        raw = self._clamp(raw, self.steer_min, self.steer_max)

        return int(raw)

    @staticmethod
    def _clamp(x, mn, mx):
        if x < mn:
            return mn
        if x > mx:
            return mx
        return x
