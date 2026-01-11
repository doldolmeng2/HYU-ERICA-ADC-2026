#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2


class RubberconeNavigator:
    """
    라바콘 주행용 라이다 처리 모듈 (ROS 노드 아님 / main에서 import해서 사용)

    외부(=main)에서 쓰는 이름/형태 유지:
      - RubberconeNavigator()
      - count_cones(scan_msg) -> int        # FOV(±60deg) 내 라바콘 개수
      - get_offset(scan_msg, enable_viz=False, show_viz=True)
          -> (offset(rad), debug(dict), viz_img(or None))

    요구사항 구현:
      1) LaserScan 점들을 10cm 기준으로 클러스터링 -> 각 클러스터에서 가장 가까운 점만 남김(라바콘 후보)
      2) 정면 기준 ±60도(FOV) 밖은 제거(버림)
      3) FOV 내 라바콘 개수 세기(모드 변경용)
      4) FOV 내에서 좌 1개/우 1개(각각 가장 가까운 것) 선택
      5) 두 라바콘 중점(mid) 계산, mid 각도와 정면(0) 각도 차이를 offset으로
         - mid가 왼쪽이면 offset 음수, 오른쪽이면 offset 양수
      6) 시각화:
         - 클러스터링된 라바콘 후보 전체 표시(각도 필터로 제거된 것도 별도 색으로)
         - 선택된 좌/우 2개 강조 표시
         - 중앙선(정면) + offset 방향선 표시
    """

    def __init__(
        self,
        cluster_dist_m: float = 0.10,
        fov_deg: float = 75.0,
        min_range_m: float = 0.05,
        max_range_m: float = 1.0,
        # 시각화
        viz_size: int = 600,
        viz_range_m: float = 1.0,
        # 좌표 보정 옵션(환경마다 라이다 장착/드라이버 방향이 다를 수 있어 옵션 제공)
        angle_offset_deg: float = 0.0,   # 모든 각도에 더할 오프셋(도)
        flip_lr: bool = False,           # 좌/우 뒤집기 (y/theta 부호 반전)
        flip_front: bool = False         # 전/후 뒤집기 (theta에 pi 추가)
    ):
        self.cluster_dist_m = float(cluster_dist_m)
        self.fov_rad = math.radians(float(fov_deg))
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)

        self.viz_size = int(viz_size)
        self.viz_range_m = float(viz_range_m)
        self.viz_win = "rubbercone_viz"

        self.angle_offset_rad = math.radians(float(angle_offset_deg))
        self.flip_lr = bool(flip_lr)
        self.flip_front = bool(flip_front)



    # -----------------------------
    # 공용 유틸
    # -----------------------------
    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _is_finite(x: float) -> bool:
        return bool(np.isfinite(x))

    # -----------------------------
    # 0) LaserScan -> 점 리스트 (x,y,r,theta)
    # -----------------------------
    def _scan_to_points(self, scan_msg):
        """
        반환: [(x, y, r, theta), ...]
          - theta: 정면 0, 좌 +, 우 -
          - x: 전방 +, y: 좌 +
        """
        pts = []

        if scan_msg is None:
            return pts

        angle = float(scan_msg.angle_min)
        inc = float(scan_msg.angle_increment)

        # angle_increment가 0이면 절대 진행 불가
        if inc == 0.0 or not self._is_finite(inc):
            return pts

        for r in scan_msg.ranges:
            # r 방어
            if r is None:
                angle += inc
                continue
            r = float(r)
            if not self._is_finite(r):
                angle += inc
                continue
            if r < self.min_range_m or r > self.max_range_m:
                angle += inc
                continue

            a = angle + self.angle_offset_rad
            if self.flip_front:
                a += math.pi
            a = self._wrap_to_pi(a)

            x = r * math.cos(a)
            y = r * math.sin(a)
            theta = a

            # 좌/우 뒤집기 옵션: y/theta 부호 반전
            if self.flip_lr:
                y = -y
                theta = -theta

            if self._is_finite(x) and self._is_finite(y) and self._is_finite(theta):
                pts.append((float(x), float(y), float(r), float(theta)))

            angle += inc

        # theta 기준 정렬(클러스터링에서 인접점 판단용)
        pts.sort(key=lambda p: p[3])
        return pts

    # -----------------------------
    # 1) 10cm 클러스터링 -> 라바콘 후보(대표점)
    # -----------------------------
    def extract_cone_candidates(self, scan_msg):
        """
        반환: cones = [{"x","y","r","theta"}, ...]
        클러스터 기준:
          - theta 정렬된 점들에서 인접 점 사이 거리 <= cluster_dist_m 이면 같은 클러스터
          - 각 클러스터에서 r 최소(가장 가까운 점)만 남김
        """
        pts = self._scan_to_points(scan_msg)
        if len(pts) == 0:
            return []

        clusters = []
        cur = [pts[0]]

        for i in range(1, len(pts)):
            x0, y0, _, _ = pts[i - 1]
            x1, y1, _, _ = pts[i]
            dist = math.hypot(x1 - x0, y1 - y0)

            if dist <= self.cluster_dist_m:
                cur.append(pts[i])
            else:
                clusters.append(cur)
                cur = [pts[i]]
        clusters.append(cur)

        cones = []
        for c in clusters:
            # 가장 가까운 점만 남김
            bx, by, br, bth = min(c, key=lambda p: p[2])
            cones.append({"x": bx, "y": by, "r": br, "theta": bth})

        return cones

    # -----------------------------
    # 2) FOV(±60도) 필터
    # -----------------------------
    def filter_fov(self, cones):
        """cones 중 |theta| <= fov_rad만 반환"""
        cones_in = [c for c in cones if abs(c["theta"]) <= self.fov_rad]
        cones_out = [c for c in cones if abs(c["theta"]) > self.fov_rad]
        return cones_in, cones_out

    # -----------------------------
    # 3) FOV 내 라바콘 개수(모드 변경용)
    # -----------------------------
    def count_cones(self, scan_msg) -> int:
        cones_all = self.extract_cone_candidates(scan_msg)
        cones_in, _ = self.filter_fov(cones_all)
        n = int(len(cones_in))
        print("rubbercone number (FOV):", n)
        return n

    # -----------------------------
    # 4) 좌 1개 / 우 1개 선택 (각각 가장 가까운 것)
    # -----------------------------
    def select_left_right_cones(self, scan_msg):
        """
        반환: (left_cone, right_cone, cones_all, cones_in, cones_out)
          - left: theta > 0 (좌)
          - right: theta < 0 (우)
          - 각 그룹에서 r 가장 작은(가장 가까운) 1개 선택
        """
        cones_all = self.extract_cone_candidates(scan_msg)
        cones_in, cones_out = self.filter_fov(cones_all)

        lefts = [c for c in cones_in if c["theta"] > 0.0]
        rights = [c for c in cones_in if c["theta"] < 0.0]

        left_cone = min(lefts, key=lambda c: c["r"]) if len(lefts) > 0 else None
        right_cone = min(rights, key=lambda c: c["r"]) if len(rights) > 0 else None

        return left_cone, right_cone, cones_all, cones_in, cones_out

    # -----------------------------
    # 시각화 유틸
    # -----------------------------
    def _xy_to_pixel(self, x, y):
        """
        라이다 좌표계:
          x: 전방(+)
          y: 좌(+)

        화면:
          - 차량은 아래 중앙
          - 위쪽이 전방
          - "좌(+y)"는 화면에서 왼쪽으로 가도록 매핑(자연스러운 좌표계)
        """
        if not self._is_finite(x) or not self._is_finite(y):
            return None

        S = self.viz_size
        rng = self.viz_range_m
        if rng <= 0.0 or not self._is_finite(rng):
            return None

        origin_px = S // 2
        origin_py = int(S * 0.9)

        scale = (S * 0.8) / rng
        if not self._is_finite(scale) or scale <= 0.0:
            return None

        # ✅ y(좌+) -> 화면 왼쪽(-)
        px = int(origin_px - (y * scale))
        py = int(origin_py - (x * scale))

        # 화면 밖이면 그리지 않도록 None 처리(시각화 안정성)
        if px < -50 or px > S + 50 or py < -50 or py > S + 50:
            return None

        return px, py

    def _draw_viz(self, cones_in, cones_out, left, right, offset, dist_mid, mid_xy, debug_text=""):
        S = self.viz_size
        img = np.zeros((S, S, 3), dtype=np.uint8)

        # 차량 위치
        car_p = self._xy_to_pixel(0.0, 0.0)
        if car_p is None:
            return img
        car_px, car_py = car_p
        cv2.circle(img, (car_px, car_py), 6, (255, 255, 255), -1)
        cv2.putText(img, "CAR", (car_px + 10, car_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 정면 중앙선(진행방향)
        fwd_p = self._xy_to_pixel(self.viz_range_m, 0.0)
        if fwd_p is not None:
            cv2.line(img, (car_px, car_py), fwd_p, (80, 80, 80), 2)

        # FOV 경계선(±60도)
        r = min(self.viz_range_m, max(0.1, self.viz_range_m))
        xL = r * math.cos(self.fov_rad)
        yL = r * math.sin(self.fov_rad)
        xR = r * math.cos(-self.fov_rad)
        yR = r * math.sin(-self.fov_rad)

        Lp = self._xy_to_pixel(xL, yL)
        Rp = self._xy_to_pixel(xR, yR)
        if Lp is not None:
            cv2.line(img, (car_px, car_py), Lp, (0, 128, 255), 2)
        if Rp is not None:
            cv2.line(img, (car_px, car_py), Rp, (0, 128, 255), 2)

        # 각도 필터로 제거된 라바콘(OUT) - 어두운 색
        for c in cones_out:
            p = self._xy_to_pixel(c["x"], c["y"])
            if p is None:
                continue
            cv2.circle(img, p, 3, (60, 60, 60), -1)

        # FOV 내 라바콘(IN) - 초록
        for c in cones_in:
            p = self._xy_to_pixel(c["x"], c["y"])
            if p is None:
                continue
            cv2.circle(img, p, 4, (0, 255, 0), -1)

        # 선택된 left/right 강조
        if left is not None:
            p = self._xy_to_pixel(left["x"], left["y"])
            if p is not None:
                cv2.circle(img, p, 10, (0, 255, 255), 2)  # 노랑 테두리
                cv2.putText(img, "L", (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if right is not None:
            p = self._xy_to_pixel(right["x"], right["y"])
            if p is not None:
                cv2.circle(img, p, 10, (255, 0, 255), 2)  # 보라 테두리
                cv2.putText(img, "R", (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # 중점 표시 + offset 방향선
        if mid_xy is not None and dist_mid is not None and self._is_finite(dist_mid):
            mx, my = mid_xy
            mp = self._xy_to_pixel(mx, my)
            if mp is not None:
                cv2.circle(img, mp, 7, (255, 255, 0), -1)  # 하늘색
                cv2.putText(img, "MID", (mp[0] + 10, mp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # CAR->MID 선
                cv2.line(img, (car_px, car_py), mp, (255, 255, 0), 2)

            # offset 방향선: dist_mid 거리만큼 offset 각도로 뻗는 점
            # offset 정의: 왼쪽 음수, 오른쪽 양수
            # 라이다 좌표에서 y(좌+)이므로, offset 부호가 반대인 점을 주의해서 계산
            # 우리는 offset = -atan2(my,mx)로 만들었으니, 시각화에서 offset 방향은
            # theta_mid = -offset 로 해석하면 됨.
            theta_mid = -offset
            tx = dist_mid * math.cos(theta_mid)
            ty = dist_mid * math.sin(theta_mid)
            tp = self._xy_to_pixel(tx, ty)
            if tp is not None:
                cv2.line(img, (car_px, car_py), tp, (0, 200, 255), 2)
                cv2.putText(img, "OFFSET", (tp[0] + 8, tp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # 텍스트
        offset_deg = math.degrees(offset) if self._is_finite(offset) else 0.0
        cv2.putText(img, f"offset(rad)={offset:+.3f}  ({offset_deg:+.1f} deg)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if debug_text:
            cv2.putText(img, debug_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        return img

    # -----------------------------
    # 5) offset 계산 + 시각화
    # -----------------------------
    def get_offset(self, scan_msg, enable_viz: bool = False, show_viz: bool = True):
        """
        offset 계산:
          - 좌 1개/우 1개가 모두 있을 때만 offset 유효
          - mid가 왼쪽이면 offset 음수, 오른쪽이면 offset 양수

        반환:
          offset(rad), debug(dict), viz_img(or None)
        """
        debug = {
            "ok": False,
            "reason": "",
            "cone_count_all": 0,
            "cone_count_fov": 0,
            "left": None,
            "right": None,
            "mid": None,          # {"x","y","dist"}
            "theta_mid": None,    # atan2(my,mx) (라이다 기준)
            "offset": 0.0
        }

        left, right, cones_all, cones_in, cones_out = self.select_left_right_cones(scan_msg)
        debug["cone_count_all"] = int(len(cones_all))
        debug["cone_count_fov"] = int(len(cones_in))
        debug["left"] = left
        debug["right"] = right

        offset = 0.0
        viz = None

        if left is None or right is None:
            debug["reason"] = "left or right cone missing in FOV"
            if enable_viz:
                viz = self._draw_viz(
                    cones_in=cones_in,
                    cones_out=cones_out,
                    left=left,
                    right=right,
                    offset=0.0,
                    dist_mid=None,
                    mid_xy=None,
                    debug_text=debug["reason"]
                )
                if show_viz:
                    cv2.imshow(self.viz_win, viz)
                    cv2.waitKey(1)
            return float(offset), debug, viz

        # mid 계산
        mx = 0.5 * (left["x"] + right["x"])
        my = 0.5 * (left["y"] + right["y"])
        dist_mid = math.hypot(mx, my)

        if not self._is_finite(dist_mid) or dist_mid <= 1e-6:
            debug["reason"] = "mid distance invalid"
            if enable_viz:
                viz = self._draw_viz(
                    cones_in=cones_in,
                    cones_out=cones_out,
                    left=left,
                    right=right,
                    offset=0.0,
                    dist_mid=None,
                    mid_xy=None,
                    debug_text=debug["reason"]
                )
                if show_viz:
                    cv2.imshow(self.viz_win, viz)
                    cv2.waitKey(1)
            return 0.0, debug, viz

        # 정면(0 rad)과 mid 방향(theta_mid)의 각도 차이가 offset
        # theta_mid: 왼쪽 +, 오른쪽 -
        theta_mid = math.atan2(my, mx)

        # 요구: 왼쪽이면 offset 음수, 오른쪽이면 양수
        offset = -theta_mid

        debug["ok"] = True
        debug["mid"] = {"x": float(mx), "y": float(my), "dist": float(dist_mid)}
        debug["theta_mid"] = float(theta_mid)
        debug["offset"] = float(offset)

        if enable_viz:
            viz = self._draw_viz(
                cones_in=cones_in,
                cones_out=cones_out,
                left=left,
                right=right,
                offset=offset,
                dist_mid=dist_mid,
                mid_xy=(mx, my),
                debug_text="OK"
            )
            if show_viz:
                cv2.imshow(self.viz_win, viz)
                cv2.waitKey(1)

        return float(offset), debug, viz
