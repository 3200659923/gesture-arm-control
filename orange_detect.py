# -*- coding: utf-8 -*-
"""
橙色物体识别测试程序
功能：打开USB摄像头，识别橙色罐子，框出位置，显示中心坐标
"""
import cv2
import numpy as np

# ===== 参数配置 =====
# 橙色 HSV 阈值（HSV颜色空间）
# 先试这个范围，不对再调
LOWER_ORANGE = np.array([5, 120, 120])
UPPER_ORANGE = np.array([25, 255, 255])

# 摄像头编号：0=内置，1=USB外接
CAMERA_ID = 1

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[错误] 打不开摄像头 {CAMERA_ID}")
        print("试试改成 0 或 2")
        return

    print("=" * 50)
    print("橙色物体识别测试")
    print("按 q 退出")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转 HSV 颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 颜色过滤：只留橙色
        mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

        # 形态学操作，去除噪点
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找最大的轮廓
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 500:  # 面积阈值，过滤小噪点
                # 画外接矩形
                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2

                # 画框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # 显示信息
                cv2.putText(frame, f"Object: ({cx}, {cy})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area:.0f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 归一化坐标（0~1），后面给机械臂用
                h_frame, w_frame = frame.shape[:2]
                norm_x = cx / w_frame
                norm_y = cy / h_frame
                cv2.putText(frame, f"Norm: ({norm_x:.2f}, {norm_y:.2f})", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 显示掩码（看颜色过滤效果）
        cv2.imshow("Mask (orange)", mask)
        cv2.imshow("Orange Detect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
