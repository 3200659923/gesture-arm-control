# -*- coding: utf-8 -*-
"""
MediaPipe 手部视觉识别 + 机械臂控制上位机
============================================
功能：
  1. USB 摄像头实时采集（也可用静态图片）
  2. MediaPipe Hands 输出 21 个手部关键点并绘制骨架
  3. 识别手掌左右 / 上下 / 远近位置
  4. 识别 5 根手指状态，判断"握拳 / 张开"
  5. 坐标映射为 4 路舵机角度：A底座 B大臂 C小臂 D夹爪
  6. 可选通过串口发送 "Axx,Bxx,Cxx,Dxx\n" 协议帧

用法：
  python hand_control.py                     # 摄像头实时识别（默认）
  python hand_control.py --camera 1          # 指定摄像头编号
  python hand_control.py --image hand.jpg    # 用静态图片识别（无摄像头时验证用）
  python hand_control.py --port COM3         # 打开串口发送控制帧
  python hand_control.py --no-window         # 无窗口模式（服务器/无显示环境）

运行环境：Python 3.12 + mediapipe 0.10.21  + pyserial
"""
import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# 一、全局配置
# ============================================================

# 4 路舵机角度限幅（度）—— 安全保护，防止撞击支架
SERVO_LIMIT = {
    "A": (5, 175),    # 底座 MG996R
    "B": (10, 170),   # 大臂 MG996R
    "C": (0, 120),    # 小臂：限到 120°，避开 135° 重力矩最大的抖动点
    "D": (0, 90),    # 夹爪
}

# 位置映射的"工作范围"（归一化坐标 0~1）
# 手左右 -> A 底座：x 从 0.15~0.85 线性映射到角度区间
X_RANGE = (0.15, 0.75)
A_RANGE = (175, 5)          # 手往左(cx小)→A大→底座左转；手往右(cx大)→A小→底座右转
# 手上下 -> B 大臂：y 从 0.10~0.9 映射（注意图像 y 轴向下）
Y_RANGE = (0.15, 0.75)       # 手上下范围：中间 50% 区域就能覆盖全部角度，不用抬很高
B_RANGE = (170, 5)         # 手抬高（y 变小）-> B=10（大臂抬起）；手放下（y 变大）-> B=170（大臂下落）
# 手掌倾斜 -> C 小臂：往前倾=前伸，往后仰=收回

C_RANGE = (90, 0)         # 手近(size大)->C前伸，手远(size小)->C收回

# 防抖参数
DEAD_ZONE = 0.05            # 死区：位置变化小于该值不更新，减少抖动
SMOOTH = 0.3              # 指数平滑系数（0~1，越小越慢越稳，越大越快越抖）
SEND_INTERVAL = 0.05        # 串口发送最小间隔（秒），约 20Hz

# 手部判定阈值：>= 5 根手指伸直算"张开"，<= 1 根算"握拳"
OPEN_THRESHOLD = 5
FIST_THRESHOLD = 2

# 可视化
WINDOW_NAME = "MediaPipe Hand Control"
COLOR_BG = (30, 30, 46)          # BGR 背景
COLOR_PANEL = (45, 45, 68)       # 信息面板
COLOR_TEXT = (235, 235, 235)
COLOR_ACCENT = (120, 200, 120)   # 绿色高亮
COLOR_WARN = (90, 150, 250)      # 蓝色


# ============================================================
# 二、上位机主控制器类
# ============================================================
class HandController:
    """封装 MediaPipe 识别 + 角度映射 + 串口发送的完整流程"""

    def __init__(self, port=None, baudrate=115200):
        # ---- MediaPipe Hands 初始化 ----
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,   # 视频流模式（连续帧跟踪，更快）
            max_num_hands=1,           
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # ---- 串口（可选）----
        self.serial = None
        if port:
            import serial  # 用到才导入，避免无串口时报错
            self.serial = serial.Serial(port, baudrate, timeout=0.1)
            print(f"[OK] 串口已打开：{port} @ {baudrate}")

        # ---- 平滑缓冲 ----
        self.smooth_angles = {k: 90 for k in SERVO_LIMIT}
        self.smooth_angles["D"] = SERVO_LIMIT["D"][0]

        self._last_pos = None      # 上一帧位置（用于死区）
        self._last_send = 0.0      # 上一次串口发送时间
        self.finger_history = []   # 手指识别历史（5帧投票）

    # --------------------------------------------------------
    # 1. 识别一帧：返回手部关键点结果
    # --------------------------------------------------------
    def detect(self, frame):
        """输入 BGR 帧，返回 MediaPipe results 对象"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False      # 提高性能
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    # --------------------------------------------------------
    # 2. 位置判定：左右 / 上下 / 远近
    # --------------------------------------------------------
    @staticmethod
    def hand_position(landmark):
        """
        根据 21 关键点计算手掌信息，返回 (cx, cy, tilt, size)：
          cx  : 手掌中心归一化 x (0~1)，越大越靠右
          cy  : 手掌中心归一化 y (0~1)，图像中越往下越大
          tilt: 手掌倾斜度（掌心y - 手腕y），正值=往前倾，负值=往后仰
          size: 手掌大小（手腕到指尖距离），越大表示离摄像头越近
        """
        wrist = landmark.landmark[0]   # 手腕
        mid   = landmark.landmark[9]   # 中指掌根（掌心）
        tip   = landmark.landmark[12]  # 中指指尖
        cx  = (wrist.x + mid.x) / 2.0
        cy  = (wrist.y + mid.y) / 2.0
        tilt = mid.y - wrist.y          # 手掌倾斜：正值=往前倾，负值=往后仰
        # 手掌大小：手腕到中指指尖的距离
        size = ((wrist.x - tip.x)**2 + (wrist.y - tip.y)**2) ** 0.5
        return cx, cy, tilt, size

    # --------------------------------------------------------
    # 3. 手势判定：逐指伸直检测 + 握拳/张开
    # --------------------------------------------------------
    @staticmethod
    def finger_state(landmark):
        """
        返回 5 个手指的伸直状态列表 [拇指, 食指, 中指, 无名指, 小指]
        1 = 伸直，0 = 弯曲。
        用指尖到 MCP 关节的距离 / 手掌长度 判断：
          - 伸直：指尖到 MCP 距离大 ≈ 手掌长度
          - 弯曲：指尖到 MCP 距离小 < 0.6 手掌长度
        """
        pts = landmark.landmark
        wrist = pts[0]

        def dist(a, b):
            return ((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) ** 0.5

        # 手掌长度基准：手腕到中指 MCP(9) 的距离
        palm_len = dist(wrist, pts[9])

        fingers = []

        # 拇指：指尖4 到 MCP(2) 的距离 / 手掌长度
        thumb_ratio = dist(pts[4], pts[2]) / palm_len
        fingers.append(1 if thumb_ratio > 0.45 else 0)

        # 其余四指：指尖到 MCP 的距离 / 手掌长度
        # 食指: MCP=5, TIP=8
        # 中指: MCP=9, TIP=12
        # 无名指: MCP=13, TIP=16
        # 小指: MCP=17, TIP=20
        fingers_data = [
            (5, 8),    # 食指
            (9, 12),   # 中指
            (13, 16),  # 无名指
            (17, 20),  # 小指
        ]
        for mcp, tip in fingers_data:
            ratio = dist(pts[tip], pts[mcp]) / palm_len
            fingers.append(1 if ratio > 0.6 else 0)

        # ---- 新增：指尖距离判断（辅助验证）----
        # 计算 5 个指尖之间的平均距离，判断整体张开还是握拳
        tips = [pts[4], pts[8], pts[12], pts[16], pts[20]]
        total_dist = 0
        count = 0
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                total_dist += dist(tips[i], tips[j])
                count += 1
        avg_dist = total_dist / count if count > 0 else 0
        # 距离大 → 张开；距离小 → 握拳
        spread_ratio = avg_dist / palm_len


        return fingers

    # --------------------------------------------------------
    # 4. 坐标映射：位置/手势 -> 4 路舵机角度（限幅 + 平滑）
    # --------------------------------------------------------
    def map_angles(self, cx, cy, tilt, size, fingers):
        # A 底座：手左右移动
        a = np.interp(cx, X_RANGE, A_RANGE)
        # B 大臂：手上下移动
        b = np.interp(cy, Y_RANGE, B_RANGE)
        # C 小臂：用手大小（远近）控制
        # 手离摄像头近（size大）→ C 前伸；手离摄像头远（size小）→ C 收回
        size_range = (0.2, 0.35)  # size 工作范围
        c = np.interp(size, size_range, [0, 100])
        # D 夹爪：线性控制 - 伸几根手指，夹爪就开多少
        n_open = sum(fingers)

        # ---- D 夹爪：抓取锁定模式 ----
        # 一旦握拳抓取，就锁定夹爪闭合；只有主动张开才松开
        if not hasattr(self, 'gripper_locked'):
            self.gripper_locked = False  # 默认未锁定

        # 用指尖距离判断张开/握拳（更稳定）
        SPREAD_TH = 0.9  # 小于0.9握拳，大于0.9张开

        if not self.gripper_locked:
            # 未锁定状态：检测到握拳就锁定
            if self.spread_ratio < SPREAD_TH:
                self.gripper_locked = True
        else:
            # 已锁定状态：检测到张开才解锁
            if self.spread_ratio > SPREAD_TH:
                self.gripper_locked = False

        if self.gripper_locked:
            # 锁定中：完全闭合（两档控制）
            d = SERVO_LIMIT["D"][1]
        else:
            # 未锁定：完全张开（两档控制）
            d = SERVO_LIMIT["D"][0]

        angles = {"A": float(a), "B": float(b), "C": float(c), "D": float(d)}

        # ---- 限幅（安全保护，防撞支架）----
        for k in angles:
            lo, hi = SERVO_LIMIT[k]
            angles[k] = max(lo, min(hi, angles[k]))

        # ---- 指数平滑（消除抖动）----
        # D 舵机是两档控制，跳过平滑，瞬间到位
        for k in angles:
            if k == "D":
                # D 舵机不平滑，直接到位
                self.smooth_angles[k] = angles[k]
            else:
                angles[k] = self.smooth_angles[k] * (1 - SMOOTH) + angles[k] * SMOOTH
                self.smooth_angles[k] = angles[k]
        return angles

    # --------------------------------------------------------
    # 5. 死区过滤：位置变化太小就不更新（进一步防抖）
    # --------------------------------------------------------
    def in_dead_zone(self, cx, cy):
        if self._last_pos is None:
            self._last_pos = (cx, cy)
            return False
        dx = abs(cx - self._last_pos[0])
        dy = abs(cy - self._last_pos[1])
        if dx < DEAD_ZONE and dy < DEAD_ZONE:
            return True
        self._last_pos = (cx, cy)
        return False

    # --------------------------------------------------------
    # 6. 串口发送协议帧
    # --------------------------------------------------------
    def send_angles(self, angles):
        if self.serial is None:
            return
        now = time.time()
        if now - self._last_send < SEND_INTERVAL:   # 限频
            return
        self._last_send = now
        msg = (f"A{int(round(angles['A']))},"
               f"B{int(round(angles['B']))},"
               f"C{int(round(angles['C']))},"
               f"D{int(round(angles['D']))}\n")
        self.serial.write(msg.encode("ascii"))

    # --------------------------------------------------------
    # 7. 单帧完整处理：识别 -> 判定 -> 映射 -> 发送 -> 绘制
    # --------------------------------------------------------
    def process_frame(self, frame, draw=True):
        """处理一帧，返回 (处理后的画面, 手势文本, 角度字典)"""
        results = self.detect(frame)
        gesture = "NO HAND"
        angles = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制骨架
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # 位置
                cx, cy, tilt, size = self.hand_position(hand_landmarks)
                # 手势（5帧投票防抖）
                fingers_raw = self.finger_state(hand_landmarks)
                # 计算 spread_ratio 存到 self
                wrist = hand_landmarks.landmark[0]
                pts = hand_landmarks.landmark
                def dist(a, b):
                    return ((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) ** 0.5
                palm_len = dist(wrist, pts[9])
                tips = [pts[4], pts[8], pts[12], pts[16], pts[20]]
                total_dist = 0
                count = 0
                for i in range(len(tips)):
                    for j in range(i+1, len(tips)):
                        total_dist += dist(tips[i], tips[j])
                        count += 1
                self.spread_ratio = total_dist / count / palm_len if count > 0 else 0
                self.finger_history.append(fingers_raw)
                if len(self.finger_history) > 3:
                    self.finger_history.pop(0)
                # 对每根手指取多数投票
                fingers = []
                for i in range(5):
                    votes = [h[i] for h in self.finger_history]
                    fingers.append(1 if votes.count(1) >= 2 else 0)
                n_open = sum(fingers)
                # 注意：OpenCV 默认字体不支持中文，画面文字统一用英文
                if n_open >= OPEN_THRESHOLD:
                    gesture = "OPEN"
                elif n_open <= FIST_THRESHOLD:
                    gesture = "FIST"
                else:
                    gesture = f"PARTIAL {n_open}/5"

                # 死区过滤 + 角度映射
                if not self.in_dead_zone(cx, cy):
                    angles = self.map_angles(cx, cy, tilt, size, fingers)
                    self.send_angles(angles)
                else:
                    # 位置在死区时，A/B/C 保持上次，D 夹爪单独更新（因为手势变了）
                    angles = dict(self.smooth_angles)
                    n_open = sum(fingers)
                    d_new = np.interp(n_open, [0, 5],
                                      [SERVO_LIMIT["D"][1], SERVO_LIMIT["D"][0]])
                    # D 也要平滑一下，不然跳太快
                    d_smooth = angles["D"] * (1 - SMOOTH) + d_new * SMOOTH
                    angles["D"] = d_smooth
                    self.smooth_angles["D"] = d_smooth
                    self.send_angles(angles)

                if draw:
                    self._draw_panel(frame, cx, cy, tilt, size, fingers, gesture, angles, getattr(self, 'gripper_locked', False))

        return frame, gesture, angles

    # --------------------------------------------------------
    # 8. 屏幕信息面板绘制
    # --------------------------------------------------------
    @staticmethod
    def _draw_panel(frame, cx, cy, tilt, size, fingers, gesture, angles, gripper_locked=False):
        h, w = frame.shape[:2]
        # 半透明信息面板
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (w - 8, 170), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        lock_text = "LOCKED" if gripper_locked else "OPEN"
        lines = [
            f"Gesture : {gesture}  [{lock_text}]",
            f"Pos     : x={cx:.2f}  y={cy:.2f}  size={size:.2f}",
            f"Fingers : {fingers}  n={sum(fingers)}",
            f"Angles  : A={angles['A']:.0f}  B={angles['B']:.0f}  "
            f"C={angles['C']:.0f}  D={angles['D']:.0f}",
            "Ctrl    : Q quit",
        ]
        for i, txt in enumerate(lines):
            color = COLOR_ACCENT if i == 0 else COLOR_TEXT
            cv2.putText(frame, txt, (20, 34 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        cv2.LINE_AA)

    # --------------------------------------------------------
    # 9. 摄像头实时主循环
    # --------------------------------------------------------
    def run_camera(self, camera_index=0, show_window=True, max_seconds=None,
                   width=640, height=480):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 {camera_index}。请检查："
                f"①摄像头是否被其他程序占用；②设备编号是否正确（可用 0/1/2 试）。")

        # 分辨率优化：1080P 跑 MediaPipe 帧率低，默认降到 640x480 提速
        # 需要高清可传 --width 1280 --height 720
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[OK] 摄像头已打开：{w}x{h}，开始实时识别。按 Q 退出...")
        fps_time = time.time()
        start = time.time()
        last_report = 0.0
        while True:
            # 定时自动退出（用于自动化验证）
            if max_seconds and (time.time() - start) > max_seconds:
                print(f"\n[OK] 已运行 {max_seconds}s，自动退出。")
                break

            ok, frame = cap.read()
            if not ok:
                print("[WARN] 读取帧失败，重试...")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)   # 镜像，操作更直观
            frame, gesture, angles = self.process_frame(frame)
            if show_window:
                cv2.imshow(WINDOW_NAME, frame)
                # 显示实时 FPS
                fps = 1.0 / (time.time() - fps_time)
                fps_time = time.time()
                cv2.putText(frame, f"FPS:{fps:.0f}", (frame.shape[1] - 95, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # 无窗口模式：约每 0.5s 打印一次状态
                if angles and (time.time() - last_report) > 0.5:
                    last_report = time.time()
                    a, b, c, d = (int(round(angles[k])) for k in "ABCD")
                    print(f"\r{gesture:8s}  A={a:3d} B={b:3d} C={c:3d} D={d:3d}",
                          end="", flush=True)

        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    # --------------------------------------------------------
    # 10. 静态图片模式（无摄像头时验证算法用）
    # --------------------------------------------------------
    def run_image(self, img_path, show_window=True):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{img_path}")
        print(f"[OK] 已加载图片：{img_path}  {img.shape[1]}x{img.shape[0]}")

        frame, gesture, angles = self.process_frame(img)
        print(f"\n===== 识别结果 =====")
        print(f"手势     : {gesture}")
        if angles:
            print(f"舵机角度 : A={angles['A']:.0f}  B={angles['B']:.0f}  "
                  f"C={angles['C']:.0f}  D={angles['D']:.0f}")
        # 保存带标注的结果图
        out_path = img_path.rsplit(".", 1)[0] + "_result.jpg"
        cv2.imwrite(out_path, frame)
        print(f"[OK] 结果已保存：{out_path}")

        if show_window:
            cv2.imshow(WINDOW_NAME, frame)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return gesture, angles


# ============================================================
# 三、主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MediaPipe 手部识别机械臂上位机")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--image", type=str, default=None,
                        help="静态图片路径（不传则使用摄像头）")
    parser.add_argument("--port", type=str, default=None,
                        help="串口名，如 COM3（不传则只识别不发送）")
    parser.add_argument("--baud", type=int, default=115200, help="串口波特率")
    parser.add_argument("--no-window", action="store_true",
                        help="无窗口模式（适合服务器/远程）")
    parser.add_argument("--seconds", type=float, default=None,
                        help="自动运行 N 秒后退出（摄像头模式），默认一直运行")
    parser.add_argument("--width", type=int, default=640,
                        help="摄像头采集宽度，默认 640（降分辨率提升帧率）")
    parser.add_argument("--height", type=int, default=480,
                        help="摄像头采集高度，默认 480")
    args = parser.parse_args()

    ctrl = HandController(port=args.port, baudrate=args.baud)
    show = not args.no_window

    try:
        if args.image:
            ctrl.run_image(args.image, show_window=show)
        else:
            ctrl.run_camera(camera_index=args.camera, show_window=show,
                            max_seconds=args.seconds,
                            width=args.width, height=args.height)
    except KeyboardInterrupt:
        print("\n[OK] 用户中断，程序退出。")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1
    finally:
        if ctrl.serial is not None:
            ctrl.serial.close()
            print("[OK] 串口已关闭。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
