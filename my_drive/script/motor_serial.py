#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import serial
import time
from std_msgs.msg import Int16MultiArray

class MotorSerialNode:
    def __init__(self):

        rospy.init_node('motor_serial_node')

        try:
            self.ser = serial.Serial('/dev/arduino', 115200, timeout=1)
            rospy.loginfo("아두이노 시리얼 연결 성공")
            
        except Exception as e:
            rospy.logerr(f"시리얼 연결 실패: {e}")
            exit(1)
            
        rospy.Subscriber("/motor", Int16MultiArray, self.motor_callback)

    def create_command(self, steer, speed):
        STX = 0xEA
        ETX = 0x03
        Length = 0x03
        d1 = 0
        d2 = 0
        
        steer = max(45, min(135, steer))
        speed = max(90, min(119, speed)) # 범위는 아두이노 설정에 맞게 조절

        # 체크섬 계산
        cs = ((~(Length + steer + speed + d1 + d2)) & 0xFF) + 1
        
        return bytearray([STX, Length, steer, speed, d1, d2, cs, ETX])

    def motor_callback(self, msg):
        try:
            if len(msg.data) >= 2:
                steer = msg.data[0]
                speed = msg.data[1]
                
                packet = self.create_command(steer, speed)
                self.ser.write(packet)
                
                
        except Exception as e:
            rospy.logerr(f"시리얼 전송 중 에러: {e}")

    def clean_up(self):
        rospy.loginfo("노드 종료: 모터 정지 명령 전송")
        try:
            # 정지 신호 (조향 90, 속도 90=정지)
            stop_packet = self.create_command(90, 90)
            self.ser.write(stop_packet)
            self.ser.close()
        except:
            pass

if __name__ == '__main__':
    node = MotorSerialNode()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.clean_up()
