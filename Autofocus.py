#!/usr/bin/env python3
# MIT License – 2019 JetsonHacks, modyfikacje 2025
# Demo: automatyczny autofocus kamery IMX219 (ArduCAM) na Jetson Nano

import cv2
import numpy as np
import subprocess
import sys

# ───────────────────────── Autofokus I²C ──────────────────────────
def set_focus(step: int) -> None:
    """
    Ustawia pozycję soczewki modułu ArduCAM IMX219-AF.
    Parametr `step` 0-1023 (0 = nieskończoność, 1023 = makro).
    """
    if not (0 <= step <= 1023):
        raise ValueError("step must be 0-1023")

    value = (step << 4) & 0x3FF0
    data1 = (value >> 8) & 0x3F      # 6 MSB
    data2 = value & 0xF0             # 4 LSB, wyrównane do bajtu
    # Bus 6, addr 0x0c – typowe dla IMX219-AF na Nano
    subprocess.run(
        ["i2cset", "-y", "6", "0x0c", str(data1), str(data2)],
        check=True
    )

# ────────────────────── miara ostrości obrazu ─────────────────────
def focus_metric(img: np.ndarray) -> float:
    """Im większa wartość, tym ostrzejszy obraz (wariant Laplace’a)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16U)
    return cv2.mean(lap)[0]

# ───────────────– GStreamer pipeline dla kamery CSI ───────────────
def gstreamer_pipeline(capture_width=1280, capture_height=720,
                       display_width=1280, display_height=720,
                       framerate=60, flip_method=0) -> str:
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        "format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

# ────────────────────────── pętla główna ───────────────────────────
def autofocus_demo() -> None:
    print("Pipeline:\n", gstreamer_pipeline())
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        sys.exit("❌ Nie mogę otworzyć kamery – sprawdź połączenia / moduł CSI")

    cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

    # parametry strojenia algorytmu
    max_step = 10       # najlepsza dotychczas pozycja
    max_metric = 0.0
    last_metric = 0.0
    dec_frames = 0
    step = 10
    finished = False

    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Brak klatki z kamery")
            break

        cv2.imshow("CSI Camera", frame)

        # tryb strojenia – dopóki obraz wyraźnie się poprawia
        if dec_frames < 6 and step < 1020:
            try:
                set_focus(step)
            except subprocess.CalledProcessError as e:
                print("⚠️  I²C error:", e)
                break

            sharp = focus_metric(frame)

            if sharp > max_metric:
                max_metric = sharp
                max_step = step

            dec_frames = dec_frames + 1 if sharp < last_metric else 0
            last_metric = sharp
            step += 10

        elif not finished:
            # ustaw najlepszą znalezioną pozycję
            set_focus(max_step)
            finished = True
            print(f"✔️  Autofokus ustawiony (step = {max_step})")

        # obsługa klawiatury
        key = cv2.waitKey(16) & 0xFF
        if key == 27:             # Esc → wyjście
            break
        elif key in (ord('r'), 10):  # r lub Enter → restart autofokusa
            max_step = 10
            max_metric = 0.0
            last_metric = 0.0
            dec_frames = 0
            step = 10
            finished = False
            print("↻ Restart autofokusa")

    cap.release()
    cv2.destroyAllWindows()

# ───────────────────────────── main ────────────────────────────────
if __name__ == "__main__":
    try:
        autofocus_demo()
    except KeyboardInterrupt:
        print("\nPrzerwano ⏹")
