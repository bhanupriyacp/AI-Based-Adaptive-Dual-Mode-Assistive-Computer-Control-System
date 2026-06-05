import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import pyautogui
import time
import math
import joblib
import numpy as np
import subprocess
import ctypes
from collections import deque



# ==========================================
# LOAD ML MODEL  — ML BASED
# ==========================================
try:
    model  = joblib.load("models/gesture_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    ML_ON  = True
    print("✅  SVM model loaded — HYBRID mode")
except Exception as e:
    ML_ON  = False
    print(f"⚠️  No model — RULE-BASED fallback ({e})")

PRED_BUFFER = deque(maxlen=8)   # majority vote — kills single-frame jitter

# ==========================================
# MEDIAPIPE
# ==========================================
mp_hands  = mp.solutions.hands
hands_det = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ==========================================
# CURSOR SETTINGS
# ==========================================
SMOOTH          = 0.10
MARGIN          = 0.22
MOVE_THRESHOLD  = 8
PINCH_THRESHOLD = 0.04

smooth_x = smooth_y = None
last_pos  = (0, 0)

# ==========================================
# CLICK SETTINGS  
# ==========================================
LEFT_DWELL   = 0.6
DOUBLE_DWELL = 1.3

last_move_time     = time.time()
has_single_clicked = False

# ==========================================
# RIGHT CLICK  
# ==========================================
RIGHT_CLICK_COOLDOWN = 1.2
last_right_click     = 0

# ==========================================
# DRAG  
# ==========================================
is_dragging = False

# ==========================================
# SCROLL  — 
# ------------------------------------------
#  ML detects "scroll" gesture.
#  Rule-based handles direction:
#    Lock neutral_y on first frame.
#    Hand ABOVE neutral → scroll UP  continuously
#    Hand BELOW neutral → scroll DOWN continuously
# ==========================================
SCROLL_DEADZONE  = 0.08
SCROLL_SPEED     = 150
SCROLL_INTERVAL  = 0.03

scroll_neutral_y = None
last_scroll_tick = 0.0

# ==========================================
# KEYBOARD   (3-fallback, no WinError 740)
# ==========================================
TABTIP            = r"C:\Program Files\Common Files\microsoft shared\ink\TabTip.exe"
osk_open          = False
last_kb_time      = 0
KEYBOARD_COOLDOWN = 2.0

def open_osk():
    global osk_open
    if os.path.exists(TABTIP):
        try:
            subprocess.Popen([TABTIP], shell=False)
            osk_open = True; print("⌨️  Touch Keyboard OPENED"); return
        except: pass
    try:
        subprocess.Popen("start /b osk", shell=True)
        osk_open = True; print("⌨️  OSK OPENED via shell"); return
    except: pass
    try:
        ctypes.windll.shell32.ShellExecuteW(
            None, "open", r"C:\Windows\System32\osk.exe", None, None, 1)
        osk_open = True; print("⌨️  OSK via ShellExecute")
    except Exception as e:
        print(f"OSK failed: {e}")

def close_osk():
    global osk_open
    for p in ("osk.exe", "TabTip.exe"):
        subprocess.call(f"taskkill /f /im {p}", shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    osk_open = False; print("⌨️  OSK CLOSED")

def toggle_osk():
    close_osk() if osk_open else open_osk()

# ==========================================
# SCREENSHOT  
# ==========================================
last_screenshot     = 0
SCREENSHOT_COOLDOWN = 3

# ==========================================
# ML PREDICTION  
# ==========================================
def predict_gesture(hand):
    wrist    = hand.landmark[0]
    features = []
    for lm in hand.landmark:
        features.append(lm.x - wrist.x)
        features.append(lm.y - wrist.y)
        features.append(lm.z - wrist.z)
    scaled = scaler.transform([features])
    raw    = model.predict(scaled)[0]
    PRED_BUFFER.append(raw)
    return max(set(PRED_BUFFER), key=PRED_BUFFER.count)

# ==========================================
# RULE-BASED FALLBACK
# (only when ML model missing)
# ==========================================
FINGER_MARGIN = 0.02
def rule_gesture(hand):
    lm = hand.landmark
    def up(t,p):   return lm[t].y < lm[p].y - FINGER_MARGIN
    def down(t,p): return lm[t].y > lm[p].y + FINGER_MARGIN
    i_up = up(8,6);   m_up = up(12,10)
    r_up = up(16,14); p_up = up(20,18)
    i_dn = down(8,6); thumb = lm[4].x < lm[3].x
    t4   = lm[4];     i8   = lm[8]
    pinch = math.hypot(t4.x-i8.x, t4.y-i8.y)

    if thumb and p_up and not i_up and not m_up and not r_up: return "keyboard_open"
    if i_up and m_up and r_up and p_up:                       return "screenshot"
    if i_up and m_up and not r_up:                            return "scroll"
    if m_up and i_dn and not r_up:                            return "right_click"
    if i_up and not m_up and not r_up and pinch < PINCH_THRESHOLD: return "drag"
    if i_up and not m_up and not r_up:                        return "move"
    return "idle"

# ==========================================
# SCROLL BAR VISUAL
# ==========================================
def draw_scroll_bar(frame, offset, h):
    bx = frame.shape[1] - 40
    cy = h // 2
    cv2.rectangle(frame, (bx, cy-120), (bx+20, cy+120), (40,40,40),  -1)
    cv2.rectangle(frame, (bx, cy-120), (bx+20, cy+120), (90,90,90),   1)
    mag = min(abs(offset)/0.15, 1.0)
    bh  = int(mag * 110)
    if offset < -SCROLL_DEADZONE:
        cv2.rectangle(frame, (bx, cy-bh), (bx+20, cy),       (50,220,90),  -1)
        cv2.putText(frame,"UP",(bx-4,cy-bh-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,220,90),1)
    elif offset > SCROLL_DEADZONE:
        cv2.rectangle(frame, (bx, cy),      (bx+20, cy+bh),  (40,150,255), -1)
        cv2.putText(frame,"DN",(bx-4,cy+bh+16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,150,255),1)
    else:
        cv2.circle(frame,(bx+10,cy),5,(180,180,180),-1)

# ==========================================
# LEGEND
# ==========================================
LEGEND = [
    ("Index only",     "Move + pinch=drag + dwell click"),
    ("Middle up",      "Right click"),
    ("Index+Middle",   "Scroll up/down"),
    ("All 4 fingers",  "Screenshot"),
    ("Thumb+Pinky",    "Toggle keyboard"),
]
def draw_legend(frame):
    y = 90
    for g,a in LEGEND:
        cv2.putText(frame, f"{g:<16} {a}", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170,215,255), 1, cv2.LINE_AA)
        y += 17

# ==========================================
# MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("=" * 50)

print("  Q to quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_det.process(rgb)
    now    = time.time()
    status = "IDLE"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # --------------------------------------------------
        # GESTURE DETECTION — ML BASED (SVM)
        # Falls back to rule-based if no model file found
        # --------------------------------------------------
        gesture = predict_gesture(hand) if ML_ON else rule_gesture(hand)

        

        # ① KEYBOARD
        if gesture == "keyboard_open":
            scroll_neutral_y = None
            status = "KEYBOARD"
            if now - last_kb_time > KEYBOARD_COOLDOWN:
                toggle_osk()
                last_kb_time = now

        # ② SCREENSHOT
        elif gesture == "screenshot":
            scroll_neutral_y = None
            status = "SCREENSHOT"
            if now - last_screenshot > SCREENSHOT_COOLDOWN:
                fname = f"shot_{int(now)}.png"
                pyautogui.screenshot(fname)
                print(f"📸  {fname}")
                last_screenshot = now

        # ③ SCROLL — neutral offset, continuous
        elif gesture == "scroll":
            curr_y = hand.landmark[8].y
            if scroll_neutral_y is None:
                scroll_neutral_y = curr_y
            offset = curr_y - scroll_neutral_y
            draw_scroll_bar(frame, offset, h)
            if abs(offset) > SCROLL_DEADZONE and now - last_scroll_tick > SCROLL_INTERVAL:
                val = int(-offset * SCROLL_SPEED)
                val = max(-15, min(15, val))
                pyautogui.scroll(val)
                last_scroll_tick = now
            status = "SCROLL UP" if offset < -SCROLL_DEADZONE else \
                     "SCROLL DOWN" if offset > SCROLL_DEADZONE else "SCROLL HOLD"

        # ④ DRAG — pinch to drag, release to drop
        elif gesture == "drag":
            scroll_neutral_y = None
            raw_x = hand.landmark[8].x
            raw_y = hand.landmark[8].y
            nx = max(0, min(1, (raw_x - MARGIN) / (1 - 2*MARGIN)))
            ny = max(0, min(1, (raw_y - MARGIN) / (1 - 2*MARGIN)))
            tx, ty = int(nx*SCREEN_W), int(ny*SCREEN_H)
            if smooth_x is None: smooth_x, smooth_y = tx, ty
            smooth_x += (tx - smooth_x) * SMOOTH
            smooth_y += (ty - smooth_y) * SMOOTH
            if not is_dragging:
                pyautogui.mouseDown()
                is_dragging = True
            pyautogui.moveTo(int(smooth_x), int(smooth_y))
            status = "DRAGGING"

        # ⑤ RIGHT CLICK
        elif gesture == "right_click":
            scroll_neutral_y = None
            if is_dragging:
                pyautogui.mouseUp(); is_dragging = False
            status = "RIGHT CLICK"
            if now - last_right_click > RIGHT_CLICK_COOLDOWN:
                pyautogui.rightClick()
                last_right_click = now

        # ⑥ MOVE + DWELL CLICK
        elif gesture == "move":
            scroll_neutral_y = None
            if is_dragging:
                pyautogui.mouseUp(); is_dragging = False

            raw_x = hand.landmark[8].x
            raw_y = hand.landmark[8].y
            nx = max(0, min(1, (raw_x - MARGIN) / (1 - 2*MARGIN)))
            ny = max(0, min(1, (raw_y - MARGIN) / (1 - 2*MARGIN)))
            tx, ty = int(nx*SCREEN_W), int(ny*SCREEN_H)

            if smooth_x is None: smooth_x, smooth_y = tx, ty
            smooth_x += (tx - smooth_x) * SMOOTH
            smooth_y += (ty - smooth_y) * SMOOTH

            dist_moved = math.hypot(smooth_x - last_pos[0], smooth_y - last_pos[1])

            if dist_moved > MOVE_THRESHOLD:
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                last_move_time     = now
                last_pos           = (smooth_x, smooth_y)
                has_single_clicked = False
                status = "MOVING"
            else:
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                dwell = now - last_move_time
                if dwell > DOUBLE_DWELL:
                    pyautogui.doubleClick()
                    last_move_time = now
                    status = "DOUBLE CLICK"
                elif dwell > LEFT_DWELL and not has_single_clicked:
                    pyautogui.click()
                    has_single_clicked = True
                    status = "LEFT CLICK"
                else:
                    status = "DWELLING"

        # ⑦ IDLE
        else:
            scroll_neutral_y = None
            last_move_time   = now
            if is_dragging:
                pyautogui.mouseUp(); is_dragging = False

    else:
        scroll_neutral_y = None
        if is_dragging:
            pyautogui.mouseUp(); is_dragging = False

    # ==========================================
    # HUD
    # ==========================================
    S_COLORS = {
        "IDLE":         (100,100,100), "MOVING":       (0,220,80),
        "DWELLING":     (0,180,255),   "LEFT CLICK":   (0,200,255),
        "DOUBLE CLICK": (0,140,255),   "RIGHT CLICK":  (60,60,230),
        "SCROLL UP":    (50,220,90),   "SCROLL DOWN":  (40,150,255),
        "SCROLL HOLD":  (160,160,160), "DRAGGING":     (220,100,255),
        "SCREENSHOT":   (255,220,0),   "KEYBOARD":     (0,230,180),
    }
    cv2.putText(frame, f"STATUS: {status}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                S_COLORS.get(status,(200,200,200)), 2, cv2.LINE_AA)

    kb_col = (0,220,100) if osk_open else (80,80,80)
    cv2.putText(frame, f"OSK:{'ON' if osk_open else 'OFF'}", (w-120,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, kb_col, 2, cv2.LINE_AA)

    mode_col = (0,200,255) if ML_ON else (200,180,0)
    cv2.putText(frame, "SVM", (w-140,h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_col, 1, cv2.LINE_AA)

    draw_legend(frame)
    cv2.imshow("Hybrid Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Cleanup
if is_dragging: pyautogui.mouseUp()
close_osk()
cap.release()
cv2.destroyAllWindows()