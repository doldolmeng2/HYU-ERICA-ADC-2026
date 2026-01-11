#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int16MultiArray 
import cv2
from cv_bridge import CvBridge
# TODO: 아래 모듈들은 네가 이후 구현할 파일들
from mask import MaskProcessor
# from stop_line import StopLineDetector
# from traffic_light import TrafficLightDetector
# from rubbercone import RubberconeNavigator
from line_offset import LineOffsetEstimator
from control import Controller

# TODO: 실제 사용하는 모터 메시지 타입에 맞춰 수정



class Modes:
    """
    너가 설계한 모드 번호를 그대로 상수로 관리.
    숫자만 써도 되지만, 이렇게 이름을 붙여두면 디버깅/가독성이 훨씬 좋아.
    """
    Y_SPLIT_RIGHT = 0   # 노란차선 갈림길 우
    Y_SPLIT_LEFT  = 1   # 노란차선 갈림길 좌
    W_LANE_FOLLOW = 2   # 흰 차선 주행
    SIGNAL_WAIT   = 3   # 신호 대기(정지)
    RUBBERCONE    = 4   # 라바콘 주행
    AR_DRIVE      = 5   # AR 주행(후 정지)


class MainNode:
    def __init__(self):
        # =========================
        # 0) ROS 노드 초기화
        # =========================
        rospy.init_node("main", anonymous=False)

        # =========================
        # 1) 파라미터(토픽 이름, 루프 주기 등)
        # =========================
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.scan_topic  = rospy.get_param("~scan_topic", "/scan")
        self.motor_topic = rospy.get_param("~motor_topic", "/motor")

        # main 루프 주기(Hz). 카메라가 30fps면 보통 30 추천.
        self.loop_hz = rospy.get_param("~loop_hz", 30)

        # =========================
        # 2) 최신 센서 데이터 버퍼(콜백에서 업데이트)
        # =========================
        self.latest_img_msg = None   # sensor_msgs/Image (ROS 메시지 원본)
        self.latest_scan_msg = None  # sensor_msgs/LaserScan

        # 이미지/라이다가 "언제 들어왔는지" 타임스탬프도 저장해두면 디버깅에 좋음
        self.last_img_time = None
        self.last_scan_time = None

        # =========================
        # 3) 모드(state) 관련 변수
        # =========================
        self.mode = Modes.Y_SPLIT_RIGHT  # 시작 모드(설계에 맞게)
        self.mode_enter_time = rospy.Time.now()  # 현재 모드에 진입한 시간(타이머 조건에 활용)

        # 흰색 정지선 감지 횟수 카운팅(설계에 있음)
        self.white_stopline_count = 0

        # “이벤트 카운트”를 안정적으로 하려면 rising edge(False->True)만 세는 게 좋음
        self.prev_stopline_detected = False

        # =========================
        # 4) 퍼블리셔 / 서브스크라이버
        # =========================
        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.cb_scan, queue_size=1)

        self.pub_motor = rospy.Publisher(self.motor_topic, Int16MultiArray, queue_size=1)

        # =========================
        # 5) 각 기능 모듈 인스턴스 생성
        #    (여기서는 뼈대만: 실제 클래스/함수는 나중에 구현)
        # =========================
        self.mask_proc = MaskProcessor()
        self.stopline_det = None
        self.tl_det = None
        self.rubber_nav = None
        self.offset_est = None
        self.controller = None

        # TODO: 네가 각 파일 구현하면 아래 주석을 풀어서 사용하면 됨
        # self.mask_proc = MaskProcessor()
        # self.stopline_det = StopLineDetector()
        # self.tl_det = TrafficLightDetector()
        # self.rubber_nav = RubberconeNavigator()
        self.offset_est = LineOffsetEstimator()
        self.controller = Controller()

        rospy.loginfo("[main] Node initialized")
        rospy.loginfo(f"[main] sub: {self.image_topic}, {self.scan_topic}")
        rospy.loginfo(f"[main] pub: {self.motor_topic}")

        # OpenCV 변환용 브릿지
        self.bridge = CvBridge()

        # 시각화 on/off 파라미터(필요하면 런치에서 끌 수 있음)
        self.enable_viz = rospy.get_param("~enable_viz", True)

        # 시각화 창 이름
        self.viz_win = "main_viz"
        if self.enable_viz:
            cv2.namedWindow(self.viz_win, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.viz_win, 50, 50)

    # =========================
    # 콜백: 이미지 수신
    # =========================
    def cb_image(self, msg: Image):
        self.latest_img_msg = msg
        self.last_img_time = rospy.Time.now()

    # =========================
    # 콜백: 라이다 수신
    # =========================
    def cb_scan(self, msg: LaserScan):
        self.latest_scan_msg = msg
        self.last_scan_time = rospy.Time.now()

    # =========================
    # 모드 변경 헬퍼
    # =========================
    def set_mode(self, new_mode: int):
        """
        모드 변경 시 공통으로 하고 싶은 동작을 여기서 처리
        - mode 변수 업데이트
        - 모드 진입 시간 갱신
        - 필요하면 모드별 변수 초기화 등
        """
        print("mode change acc")
        if new_mode == self.mode:
            return

        rospy.loginfo(f"[main] MODE CHANGE: {self.mode} -> {new_mode}")
        self.mode = new_mode
        self.mode_enter_time = rospy.Time.now()

        # 모드 진입 시 초기화가 필요하면 여기에 추가
        # 예: 신호대기 모드 들어가면 일단 정지 명령을 강제로 내린다 등

    # =========================
    # 센서 준비 여부 체크
    # =========================
    def sensors_ready(self) -> bool:
        """
        주행 로직을 돌리기 전에 최소한 이미지/라이다가 들어왔는지 확인.
        둘 다 필요한 모드가 있고, 하나만 필요한 모드도 있지만
        일단 main에서는 안전하게 둘 다 준비될 때 시작하도록 구성.
        """
        if self.latest_img_msg is None:
            return False
        if self.latest_scan_msg is None:
            return False
        return True

    # =========================
    # stopline “이벤트” 카운트 (rising edge)
    # =========================
    def update_stopline_counter(self, stopline_detected: bool):
        """
        stopline_detected가 False->True로 바뀌는 순간만 이벤트로 카운트.
        이렇게 해야 '한 정지선을 여러 프레임에서 계속 True로 잡는' 상황에서
        카운트가 폭증하는 걸 막을 수 있음.
        """
        if (not self.prev_stopline_detected) and stopline_detected:
            # rising edge 발생
            self.white_stopline_count += 1
            rospy.loginfo(f"[main] White stopline event! count={self.white_stopline_count}")

        self.prev_stopline_detected = stopline_detected

    # =========================
    # 모드 숫자를 “글자”로 바꾸는 함수
    # =========================
    def mode_to_str(self, mode: int) -> str:
        if mode == Modes.Y_SPLIT_RIGHT:
            return "0: Y_SPLIT_RIGHT"
        elif mode == Modes.Y_SPLIT_LEFT:
            return "1: Y_SPLIT_LEFT"
        elif mode == Modes.W_LANE_FOLLOW:
            return "2: W_LANE_FOLLOW"
        elif mode == Modes.SIGNAL_WAIT:
            return "3: SIGNAL_WAIT"
        elif mode == Modes.RUBBERCONE:
            return "4: RUBBERCONE"
        elif mode == Modes.AR_DRIVE:
            return "5: AR_DRIVE"
        return f"UNKNOWN({mode})"

    # =========================
    # 모드 전이 조건 (뼈대)
    # =========================
    def update_mode_transitions(self,
                                yellow_stopline: bool,
                                white_stopline: bool,
                                green_light: bool,
                                rubbercone_found: bool,
                                white_pixels_many: bool,
                                ar_found: bool):
        """
        여기서 네가 적어준 FLOW를 기반으로 모드 전이 조건을 관리.
        지금은 '조건의 자리'만 만들어두고, 실제 판정 로직은 나중에 채우면 됨.

        인자들은 모두 "판정 결과(bool)" 라고 가정.
        - yellow_stopline: 노란 정지선 감지
        - white_stopline: 흰 정지선 감지
        - green_light: 초록불 감지
        - rubbercone_found: 라바콘 인식 여부
        - white_pixels_many: 흰 픽셀이 충분히 많아 흰 차선 주행 모드로 가야 함
        - ar_found: AR 태그 인식 여부
        """

        # -------------------------
        # 0/1: 노란 갈림길 주행 중
        # -------------------------
        if self.mode in (Modes.Y_SPLIT_RIGHT, Modes.Y_SPLIT_LEFT):
            # 노란 정지선 감지 -> 신호대기(정지)
            if yellow_stopline:
                print("yello stopline detect")
                self.set_mode(Modes.SIGNAL_WAIT)
                return

        # -------------------------
        # 3: 신호 대기
        # -------------------------
        if self.mode == Modes.SIGNAL_WAIT:
            # 초록불 감지 -> 이전 갈림길 모드로 복귀(우/좌 유지)
            # (여기서는 예시로 '직전 갈림길 방향을 기억'하는 구조가 필요할 수 있음)
            if green_light:
                # TODO: 실제로는 "마지막 갈림길 방향"을 기억했다가 그걸로 복귀하는 게 안전
                # 임시로 우회전으로 복귀(나중에 수정)
                print("mode change acc2")
                self.set_mode(Modes.Y_SPLIT_RIGHT)
                return

        # -------------------------
        # 2: 흰 차선 주행
        # -------------------------
        if self.mode == Modes.W_LANE_FOLLOW:
            # 라바콘 인식 -> 라바콘 주행
            if rubbercone_found:
                print("mode change acc3")
                self.set_mode(Modes.RUBBERCONE)
                return

            # 흰 정지선 감지 -> 이벤트 카운트 증가(예: 2번 감지 시 갈림길 좌로)
            if white_stopline:
                # stopline 이벤트 카운트는 update_stopline_counter에서 하거나,
                # 여기서 직접 할 수도 있음. 지금은 자리만.
                # self.update_stopline_counter(True)
                pass

            # 흰 정지선 2번 감지 -> 갈림길 좌 모드로
            if self.white_stopline_count >= 2:
                print("mode change acc4")
                self.set_mode(Modes.Y_SPLIT_LEFT)
                return

        # -------------------------
        # 4: 라바콘 주행
        # -------------------------
        if self.mode == Modes.RUBBERCONE:
            # 라바콘이 더 이상 안 보이면 -> 흰 차선 주행으로 복귀
            if not rubbercone_found:
                print("mode change acc5")
                self.set_mode(Modes.W_LANE_FOLLOW)
                return

        # -------------------------
        # (예시) "흰 픽셀 다수 감지" -> 흰 차선 주행 전환
        # -------------------------
        # 실제로는 갈림길 이후 "흰 차선으로 넘어가는 시점"을 잡기 위해 쓰는 조건
        if self.mode in (Modes.Y_SPLIT_RIGHT, Modes.Y_SPLIT_LEFT):
            if white_pixels_many:
                print("mode change acc6")
                self.set_mode(Modes.W_LANE_FOLLOW)
                return

        # -------------------------
        # 5: AR 주행
        # -------------------------
        if self.mode == Modes.AR_DRIVE:
            # AR 태그 인식 -> AR 주행 후 정지 (여기서는 정지 모드가 따로 없으니 control에서 처리하거나)
            if ar_found:
                # TODO: “정지” 모드가 필요하면 별도 모드 추가하거나 SIGNAL_WAIT를 재활용하면 됨
                rospy.loginfo("[main] AR found -> TODO: stop after AR drive")
                return

        # -------------------------
        # (예시) 갈림길 좌에서 노란 정지선 감지 후 신호 대기, 초록불 후 다시 좌 갈림길 등
        # 위 로직들이 이미 그 흐름을 커버하도록 설계할 수 있음
        # -------------------------

    def shutdown(self):
        msg = Int16MultiArray()
        msg.data = [90, 90]
        self.pub_motor.publish(msg)
    # =========================
    # 메인 루프
    # =========================
    def run(self):
        rate = rospy.Rate(self.loop_hz)

        while not rospy.is_shutdown():
            # 센서가 아직 준비되지 않았으면 기다림
            if not self.sensors_ready():
                rospy.logwarn_throttle(1.0, "[main] Waiting for sensors... (image/scan)")
                rate.sleep()
                continue

            # ============================================
            # 1) 최신 센서 데이터 가져오기
            # ============================================
            img_msg = self.latest_img_msg
            scan_msg = self.latest_scan_msg

            # TODO: ROS Image -> cv2 image 변환은 mask.py에서 처리하거나 여기서 처리해도 됨
            # 지금은 "뼈대"라서 실제 변환/처리는 아래에서 전부 TODO 처리

            # ============================================
            # 2) 각 인식/판정 결과 만들기 (일단 자리만)
            # ============================================
            yellow_stopline = False
            white_stopline = False
            green_light = False
            rubbercone_found = False
            white_pixels_many = False
            ar_found = False

            # ROS Image -> cv2(BGR)
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

            # mask 처리
            yellow_mask, white_mask, gray_label = self.mask_proc.process(cv_img)
            
            # 시각화 
            cv2.imshow("masked image", gray_label)

            # (예시) 정지선 판정
            # if self.stopline_det is not None and masked_img is not None:
            #     if self.mode in (Modes.Y_SPLIT_RIGHT, Modes.Y_SPLIT_LEFT):
            #         yellow_stopline = self.stopline_det.detect(masked_img, mode=self.mode)
            #     elif self.mode == Modes.W_LANE_FOLLOW:
            #         white_stopline = self.stopline_det.detect(masked_img, mode=self.mode)

            # (예시) 신호등 판정
            # if self.tl_det is not None:
            #     green_light = self.tl_det.detect_green(img_msg)

            # (예시) 라바콘 판정(라이다 기반)
            # if self.rubber_nav is not None:
            #     rubbercone_found = self.rubber_nav.is_cone_found(scan_msg)

            # 흰 픽셀 다수 판정(갈림길에서 흰차선으로 넘어가는 트리거)
            white_pixels_many = self.mask_proc.white_pixels_many(gray_label, thresh=7000)


            # (예시) AR 판정(구현 예정)
            # ar_found = False

            # ============================================
            # 3) stopline 이벤트 카운트(흰 정지선)
            #    (흰 정지선 이벤트는 주행 모드에서만 카운트한다고 가정)
            # ============================================
            if self.mode == Modes.W_LANE_FOLLOW:
                # 실제 white_stopline을 계산한 다음에 넣어야 함
                self.update_stopline_counter(white_stopline)
            else:
                # 다른 모드에서는 이전 상태 초기화(원치 않으면 제거)
                self.prev_stopline_detected = False

            # ============================================
            # 4) 모드 전이 업데이트
            # ============================================
            self.update_mode_transitions(
                yellow_stopline=yellow_stopline,
                white_stopline=white_stopline,
                green_light=green_light,
                rubbercone_found=rubbercone_found,
                white_pixels_many=white_pixels_many,
                ar_found=ar_found
            )

            # ============================================
            # 5) offset 계산 (자리만)
            # ============================================
            offset = 0.0
            debug_offset = {}
            offset_viz = None

            if self.mode == Modes.RUBBERCONE and self.rubber_nav is not None:
                offset = self.rubber_nav.get_offset(scan_msg)
            else:
                if self.offset_est is not None and gray_label is not None:
                    offset, debug_offset, offset_viz = self.offset_est.get_offset(
                        gray_label=gray_label,
                        mode=self.mode,
                        frame_bgr=cv_img,        
                        enable_viz=True         
                    )

            # ============================================
            # 6) control 계산 (steer/speed) (자리만)
            # ============================================
            steer = 90   # TODO:  기준 중립값 예시
            speed = 90    # 기본은 안전하게 90

            if self.controller is not None:
                steer, speed = self.controller.compute(offset, mode=self.mode)

            # ============================================
            # 7) publish (/motor)
            # ============================================
            msg = Int16MultiArray()
            msg.data = [int(steer), int(speed)]
            self.pub_motor.publish(msg)
            
            # ============================================
            # 8) 시각화 (raw 이미지에 정보 오버레이)
            # ============================================


            if self.enable_viz:
                try:
                    # ROS Image -> cv2 BGR 이미지 변환
                    cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

                    # 화면에 표시할 텍스트 만들기
                    mode_str = self.mode_to_str(self.mode)
                    text_lines = [
                        f"MODE : {mode_str}",
                        f"offset: {offset:.3f}",
                        f"steer : {steer}",
                        f"speed : {speed}",
                        f"stopline_count: {self.white_stopline_count}",
                    ]

                    # 글자 배경 박스(가독성용) + 텍스트 그리기
                    # 위치/크기는 필요하면 조정
                    x, y = 20, 30
                    line_h = 28

                    # 텍스트 영역 높이 계산
                    box_h = line_h * len(text_lines) + 20
                    box_w = 420  # 대충 고정 폭 (더 길면 늘려도 됨)

                    # 반투명 박스(검정) 그리기: 원본에 덮어쓰기
                    overlay = cv_img.copy()
                    cv2.rectangle(overlay, (x - 10, y - 25), (x - 10 + box_w, y - 25 + box_h), (0, 0, 0), -1)
                    alpha = 0.5
                    cv_img = cv2.addWeighted(overlay, alpha, cv_img, 1 - alpha, 0)

                    # 텍스트 출력
                    for i, line in enumerate(text_lines):
                        yy = y + i * line_h
                        cv2.putText(
                            cv_img,
                            line,
                            (x, yy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                    # 창 띄우기
                    cv2.imshow(self.viz_win, cv_img)

                    # q 누르면 안전 종료
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        rospy.signal_shutdown("user quit (viz)")
                        break

                    if offset_viz is not None:
                        cv2.imshow("offset_viz", offset_viz)
                    else:
                        cv2.imshow("offset_viz", cv_img)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        rospy.signal_shutdown("user quit (viz)")
                        break

                except Exception as e:
                    rospy.logwarn_throttle(1.0, f"[main] viz error: {e}")
            # 지금은 뼈대라서 publish 생략/로그만
            rospy.loginfo_throttle(1.0, f"[main] mode={self.mode}, offset={offset:.2f}, steer={steer}, speed={speed}")

            rate.sleep()
            
        if self.enable_viz:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    node = None
    try:
        node = MainNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if node is not None:
            node.shutdown()
