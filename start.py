# -*- coding: utf-8 -*-
"""
机械臂项目启动器
功能：
  - 自动扫描所有 COM 口
  - 交互式选择串口号、摄像头号
  - 菜单选择要运行的程序
  - 自动记住上次配置
"""
import os
import sys
import json
import subprocess

CONFIG_FILE = "launcher_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def list_com_ports():
    """列出所有可用COM口"""
    ports = []
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            ports.append((p.device, p.description))
    except Exception as e:
        print(f"[警告] 无法扫描COM口: {e}")
    return ports

def list_cameras(max_test=5):
    """测试哪些摄像头编号可用"""
    import cv2
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def main():
    cfg = load_config()

    print("=" * 60)
    print("  机械臂手势控制系统 - 启动器")
    print("=" * 60)
    print()

    # 1. 选择 COM 口
    print("【1】扫描串口设备...")
    ports = list_com_ports()
    if not ports:
        print("  没找到任何 COM 口！")
        com = input("  手动输入 COM 口（如 COM20）: ").strip()
    else:
        print("  发现以下串口：")
        for i, (name, desc) in enumerate(ports):
            print(f"    [{i+1}] {name}  -  {desc}")
        # 默认选上次的
        default_com = cfg.get("com", "")
        default_idx = 0
        for i, (name, _) in enumerate(ports):
            if name == default_com:
                default_idx = i
                break
        print(f"  (上次使用: {default_com or '无'})")
        choice = input(f"  选择串口 [默认 {default_idx+1}]: ").strip()
        if choice == "":
            com = ports[default_idx][0]
        else:
            try:
                idx = int(choice) - 1
                com = ports[idx][0]
            except:
                com = ports[default_idx][0]
    print(f"  → 选中: {com}")
    print()

    # 2. 选择摄像头
    print("【2】扫描摄像头...")
    cameras = list_cameras()
    if not cameras:
        print("  没找到摄像头！")
        cam = 0
    else:
        print("  发现以下摄像头：")
        for i, c in enumerate(cameras):
            print(f"    [{i+1}] 摄像头 #{c}")
        default_cam = cfg.get("camera", 1)
        default_idx = cameras.index(default_cam) if default_cam in cameras else 0
        print(f"  (上次使用: 摄像头 #{default_cam})")
        choice = input(f"  选择摄像头 [默认 {default_idx+1}]: ").strip()
        if choice == "":
            cam = cameras[default_idx]
        else:
            try:
                idx = int(choice) - 1
                cam = cameras[idx]
            except:
                cam = cameras[default_idx]
    print(f"  → 选中: 摄像头 #{cam}")
    print()

    # 3. 选择功能
    print("【3】选择要运行的功能：")
    print("  [1] 手势控制机械臂（主程序）")
    print("  [2] 橙色物体识别测试")
    print("  [3] 坐标标定程序")
    print("  [4] 只识别不发送（测试用）")
    print("  [0] 退出")
    choice = input("  请选择 [1]: ").strip() or "1"

    # 保存配置
    cfg["com"] = com
    cfg["camera"] = cam
    save_config(cfg)

    # 4. 运行对应程序
    python_exe = ".venv\\Scripts\\python.exe"
    if not os.path.exists(python_exe):
        python_exe = "python"

    if choice == "1":
        script = "hand_control.py"
        cmd = [python_exe, script, "--port", com, "--camera", str(cam), "--width", "640", "--height", "480"]
    elif choice == "2":
        script = "orange_detect.py"
        # orange_detect.py 里摄像头号是写死的，需要临时改
        print("  → 启动橙色识别测试...")
        cmd = [python_exe, script]
    elif choice == "3":
        script = "calibrate.py"
        # calibrate.py 里 COM 口写死了，临时改
        print("  → 启动标定程序...")
        # 用环境变量传参更简单，这里直接修改临时文件
        cmd = [python_exe, script]
    elif choice == "4":
        script = "hand_control.py"
        cmd = [python_exe, script, "--camera", str(cam), "--width", "640", "--height", "480"]
    else:
        print("再见！")
        return

    print()
    print("=" * 60)
    print(f"  启动: {script}")
    print(f"  串口: {com}")
    print(f"  摄像头: #{cam}")
    print("=" * 60)
    print()

    try:
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[错误] 运行失败: {e}")

    input("\n按回车退出...")

if __name__ == "__main__":
    main()
