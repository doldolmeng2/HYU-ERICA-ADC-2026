#!/usr/bin/env python3
import math
import numpy as np
import cv2


class RubberconeNavigator:
    """
    라바콘 주행용 라이다 처리 모듈 (ROS 노드 아님 / main에서 import해서 사용)

    시각화:
    - OpenCV 창 1개에 라이다 평면(top-down)을 그려줌
    - 라바콘 후보들, 선택된 좌/우 2개, ±60도 FOV, 중점, offset 표시
    """

    def __init__(
        self,
        cluster_dist_m: float = 0.10,
        fov_deg: float = 60.0,
        min_range_m: float = 0.05,
        max_range_m: float = 1.0,
        viz_size: int = 600,         # 시각화 이미지 크기(정사각형)
        viz_range_m: float = 1.0     # 화면에 보여줄 최대 거리(m)
    ):
        self.cluster_dist_m = float(cluster_dist_m)
        self.fov_rad = math.radians(float(fov_deg))
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)

        # 시각화 설정
        self.viz_size = int(viz_size)
        self.viz_range_m = float(viz_range_m)  # 화면 끝이 이 거리

        # 창 이름 고정
        self.viz_win = "rubbercone_viz"

    # -----------------------------
    # 0) LaserScan -> (x,y,r,theta)
    # -----------------------------
    def _wrap_to_pi(self, a: float) -> float:
        """각도를 [-pi, +pi] 범위로 정규화"""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _scan_to_points(self, scan_msg):
        pts = []
        angle = scan_msg.angle_min

        for r in scan_msg.ranges:
            if r is None or math.isinf(r) or math.isnan(r):
                angle += scan_msg.angle_increment
                continue
            if r < self.min_range_m or r > self.max_range_m:
                angle += scan_msg.angle_increment
                continue

            # ✅ 180도 뒤집힘 보정: angle에 pi 더하기
            a = angle + math.pi

            # ✅ -pi~pi로 wrap (FOV 필터 / 좌우 판정 안정화)
            a = self._wrap_to_pi(a)

            x = r * math.cos(a)   # 전방 x
            y = r * math.sin(a)   # 좌측 y

            pts.append((x, y, float(r), float(a)))
            angle += scan_msg.angle_increment

        pts.sort(key=lambda p: p[3])  # theta 기준 정렬
        return pts

    # -----------------------------
    # 1) 10cm 클러스터링 -> 대표점
    # -----------------------------
    def extract_cone_candidates(self, scan_msg):
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
            best = min(c, key=lambda p: p[2])  # range 최소
            bx, by, br, ba = best
            cones.append({"x": bx, "y": by, "r": br, "theta": ba})
        return cones

    # -----------------------------
    # 2) 라바콘 개수
    # -----------------------------
    def count_cones(self, scan_msg) -> int:
        print("rubbercone number :", int(len(self.extract_cone_candidates(scan_msg))))
        return int(len(self.extract_cone_candidates(scan_msg)))

    # -----------------------------
    # 3) FOV(±60도) 내 좌/우 1개씩
    # -----------------------------
    def select_left_right_cones(self, scan_msg):
        cones = self.extract_cone_candidates(scan_msg)
        if len(cones) == 0:
            return None, None, cones

        cones_fov = [c for c in cones if abs(c["theta"]) <= self.fov_rad]

        lefts = [c for c in cones_fov if c["theta"] > 0.0]
        rights = [c for c in cones_fov if c["theta"] < 0.0]

        left_cone = min(lefts, key=lambda c: c["r"]) if len(lefts) > 0 else None
        right_cone = min(rights, key=lambda c: c["r"]) if len(rights) > 0 else None

        return left_cone, right_cone, cones

    # -----------------------------
    # 시각화 도우미
    # -----------------------------
    def _xy_to_pixel(self, x, y):
        """
        라이다 좌표계:
          x: 전방(+)
          y: 좌(+)
        시각화:
          이미지 중앙 아래쪽을 차량 위치로 두고, 위쪽이 전방이 되게 그림.
        """
        S = self.viz_size
        rng = self.viz_range_m

        # 차량 위치: 화면 아래 중앙
        origin_px = S // 2
        origin_py = int(S * 0.9)

        # m -> px 스케일
        scale = (S * 0.8) / rng  # 화면 80%를 viz_range_m로 사용

        px = int(origin_px + (y * scale))    # y(좌+) -> 화면 오른쪽(+)로
        py = int(origin_py - (x * scale))    # x(전방+) -> 화면 위쪽(-)로
        return px, py

    def _draw_viz(self, cones, left, right, offset, debug_text=""):
        S = self.viz_size
        img = np.zeros((S, S, 3), dtype=np.uint8)

        # 차량 위치
        car_px, car_py = self._xy_to_pixel(0.0, 0.0)
        cv2.circle(img, (car_px, car_py), 6, (255, 255, 255), -1)
        cv2.putText(img, "CAR", (car_px + 10, car_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 전방 방향선(정면, theta=0)
        fwd_px, fwd_py = self._xy_to_pixel(self.viz_range_m, 0.0)
        cv2.line(img, (car_px, car_py), (fwd_px, fwd_py), (80, 80, 80), 2)

        # ±FOV 경계선(좌/우 60도)
        # theta는 라이다 기준(전방 0, 좌+)
        # 전방 거리 = viz_range_m
        r = self.viz_range_m
        xL = r * math.cos(self.fov_rad)
        yL = r * math.sin(self.fov_rad)    # 좌(+)
        xR = r * math.cos(-self.fov_rad)
        yR = r * math.sin(-self.fov_rad)   # 우(-)

        Lpx, Lpy = self._xy_to_pixel(xL, yL)
        Rpx, Rpy = self._xy_to_pixel(xR, yR)
        cv2.line(img, (car_px, car_py), (Lpx, Lpy), (0, 128, 255), 2)
        cv2.line(img, (car_px, car_py), (Rpx, Rpy), (0, 128, 255), 2)
        cv2.putText(img, "+60deg", (Lpx - 80, Lpy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        cv2.putText(img, "-60deg", (Rpx + 10, Rpy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

        # 라바콘 후보들 표시(초록)
        for c in cones:
            px, py = self._xy_to_pixel(c["x"], c["y"])
            cv2.circle(img, (px, py), 4, (0, 255, 0), -1)

        # 선택된 left/right 표시(크게)
        if left is not None:
            px, py = self._xy_to_pixel(left["x"], left["y"])
            cv2.circle(img, (px, py), 9, (0, 255, 255), 2)  # 노랑 테두리
            cv2.putText(img, "L", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if right is not None:
            px, py = self._xy_to_pixel(right["x"], right["y"])
            cv2.circle(img, (px, py), 9, (255, 0, 255), 2)  # 보라 테두리
            cv2.putText(img, "R", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # 중점 및 offset 방향 표시
        if left is not None and right is not None:
            mx = 0.5 * (left["x"] + right["x"])
            my = 0.5 * (left["y"] + right["y"])
            mpx, mpy = self._xy_to_pixel(mx, my)
            cv2.circle(img, (mpx, mpy), 7, (255, 255, 0), -1)  # 하늘색 점
            cv2.putText(img, "MID", (mpx + 10, mpy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 차량->중점 선
            cv2.line(img, (car_px, car_py), (mpx, mpy), (255, 255, 0), 2)

        # offset 텍스트
        # offset은 rad 단위로 반환 (조향용으로는 나중에 scale)
        offset_deg = math.degrees(offset)
        cv2.putText(img, f"offset(rad)={offset:+.3f}  ({offset_deg:+.1f} deg)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if debug_text:
            cv2.putText(img, debug_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        return img

    # -----------------------------
    # 4) offset 계산 + 시각화
    # -----------------------------
    def get_offset(self, scan_msg, enable_viz: bool = False, show_viz: bool = True):
        """
        offset 계산 + (옵션) 시각화

        반환:
          offset(rad), debug(dict), viz_img(or None)
        """
        debug = {
            "ok": False,
            "reason": "",
            "cone_count": 0,
            "left": None,
            "right": None,
            "mid": None,
            "theta_mid": None,
            "offset": 0.0
        }

        left, right, cones_all = self.select_left_right_cones(scan_msg)
        debug["cone_count"] = len(cones_all)
        debug["left"] = left
        debug["right"] = right

        if left is None or right is None:
            debug["reason"] = "left or right cone missing in FOV"
            offset = 0.0

            viz = None
            if enable_viz:
                viz = self._draw_viz(cones_all, left, right, offset, debug_text=debug["reason"])
                if show_viz:
                    cv2.imshow(self.viz_win, viz)
                    cv2.waitKey(1)
            return offset, debug, viz

        # 중점
        mx = 0.5 * (left["x"] + right["x"])
        my = 0.5 * (left["y"] + right["y"])
        dist_mid = math.hypot(mx, my)

        theta_mid = math.atan2(my, mx)  # 좌:+, 우:-

        # 요구: 왼쪽이면 음수, 오른쪽이면 양수 -> 부호 반전
        offset = -theta_mid

        debug["ok"] = True
        debug["mid"] = {"x": mx, "y": my, "dist": dist_mid}
        debug["theta_mid"] = theta_mid
        debug["offset"] = offset

        viz = None
        if enable_viz:
            viz = self._draw_viz(cones_all, left, right, offset, debug_text="OK")
            if show_viz:
                cv2.imshow(self.viz_win, viz)
                cv2.waitKey(1)

        return float(offset), debug, viz
