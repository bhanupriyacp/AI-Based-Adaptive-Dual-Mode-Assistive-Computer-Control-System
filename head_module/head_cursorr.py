import cv2
import mediapipe as mp
import pyautogui
import time
import math
import csv
import joblib  
import numpy as np
from sklearn import svm 
pyautogui.FAILSAFE = False
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pynput.keyboard import Key, Controller
keyboard_ctrl = Controller()

# ================= SETTINGS =================
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTH = 0.07
CURSOR_GAIN = 0.5
CURSOR_DEADZONE = 0.005
pyautogui.PAUSE = 0
SCROLL_COOLDOWN = 0.05
HEAD_SCROLL_TH = 0.04

EAR_CLOSE = 0.21
EAR_OPEN  = 0.27

last_click_time = 0
CLICK_COOLDOWN = 0.8
DOUBLE_BLINK_GAP = 0.6

SMILE_PAUSE_DELTA = 0.10
SMILE_MIN_DURATION = 0.8
CALIB_TIME = 3
last_keyboard_time = 0
KEYBOARD_COOLDOWN = 5
# ============================================

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

cx, cy = SCREEN_W//2, SCREEN_H//2
nose_base_x = None
nose_base_y = None
start_time = time.time()

cursor_paused = False
smile_base = None
smile_active = False
smile_start_time = None
scroll_origin_y = None

# Eye variables
left_eye_closed = False
left_eye_start = None
blink_counter_left = 0
last_left_blink_time = 0
blink_start = None
blink_count = 0
last_blink_time = 0

blink_count_both = 0
eye_closed_both = False
blink_start_both = 0
last_screenshot = 0

last_scroll = 0

# LOAD TRAINED MODEL
try:
    svm_model = joblib.load("gesture_svm_model.pkl")
    GESTURE_NAMES = {0: "neutral",
        1: "click",
        2: "double_click",
        3: "scroll_up",
        4: "scroll_down",
        5: "smile",
        6: "screenshot"}
except:
    svm_model = None

def svm_predict(features):
    if svm_model is not None:
        # Features array  predict 
        input_data = np.array(features).reshape(1, -1)
        res = svm_model.predict(input_data)
        return GESTURE_NAMES.get(res[0], "neutral")
    return "No Model"

# ===== HELPER FUNCTIONS =====
def dist(a,b):
    return math.hypot(a.x-b.x, a.y-b.y)

def eye_ratio(lm,l,r,u,d):
    horizontal = dist(lm[l], lm[r])
    if horizontal == 0:
        return 0
    return dist(lm[u], lm[d]) / horizontal

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = face.process(rgb)
    now = time.time()

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        nose = lm[1]

        # ===== CALIBRATION =====
        if now - start_time < CALIB_TIME:
            nose_base_x = nose.x if nose_base_x is None else 0.9*nose_base_x+0.1*nose.x
            nose_base_y = nose.y if nose_base_y is None else 0.9*nose_base_y+0.1*nose.y
        else:
            if nose_base_x is not None and nose_base_y is not None:
                dx = nose.x - nose_base_x
                dy = nose.y - nose_base_y

            if abs(dx) < CURSOR_DEADZONE: dx = 0
            if abs(dy) < CURSOR_DEADZONE: dy = 0

            tx = cx + dx * SCREEN_W * CURSOR_GAIN
            ty = cy + dy * SCREEN_H * CURSOR_GAIN

            cx += SMOOTH*(tx-cx)
            cy += SMOOTH*(ty-cy)

            cx = max(0,min(SCREEN_W,cx))
            cy = max(0,min(SCREEN_H,cy))

            # Move cursor only if not paused
            if not cursor_paused:
                pyautogui.moveTo(cx,cy,_pause=False)

            # ===== SCROLL WHEN PAUSED =====
            if cursor_paused:
                if scroll_origin_y is None:
                    scroll_origin_y = nose.y

                scroll_dy = nose.y - scroll_origin_y

                if now - last_scroll > SCROLL_COOLDOWN:
                    if scroll_dy > HEAD_SCROLL_TH:
                        pyautogui.scroll(-60)
                        last_scroll = now
                    elif scroll_dy < -HEAD_SCROLL_TH:
                        pyautogui.scroll(60)
                        last_scroll = now
            else:
                scroll_origin_y = None

        # ===== EYE RATIOS =====
        left_ear  = eye_ratio(lm,33,133,159,145)
        right_ear = eye_ratio(lm,362,263,386,374)

        

        # ===== BOTH EYES 4 BLINKS → SCREENSHOT =====
        both_closed = (left_ear < EAR_CLOSE and right_ear < EAR_CLOSE)

        if both_closed:
            if not eye_closed_both:
                eye_closed_both = True
                blink_start_both = now
        else:
            if eye_closed_both:
                if 0.08 < now - blink_start_both < 0.5:
                    blink_count_both += 1
                eye_closed_both = False

        if blink_count_both >= 4 and now - last_screenshot > 4:
            pyautogui.screenshot(f"screenshot_{int(now)}.png")
            blink_count_both = 0
            last_screenshot = now

        # ===== SMILE TOGGLE PAUSE =====
        mouth_v = dist(lm[13], lm[14])
        mouth_h = dist(lm[78], lm[308])
        smile_ratio = mouth_v / mouth_h if mouth_h > 0 else 0

        if now - start_time > CALIB_TIME and smile_base is None:
            smile_base = smile_ratio

        if smile_base is not None:
            if smile_ratio - smile_base > SMILE_PAUSE_DELTA:
                if not smile_active:
                    smile_active = True
                    smile_start_time = now
            else:
                if smile_active:
                    smile_active = False
                    if now - smile_start_time >= SMILE_MIN_DURATION:
                        cursor_paused = not cursor_paused

        # ORIGINAL LOGIC RETAINED - JUST CALLING PREDICT
        features = [nose.x, nose.y, left_ear, right_ear, smile_ratio]
       
        gesture_label = svm_predict(features)
        now = time.time()
        # ===== SMART BLINK SYSTEM =====
        if left_ear < EAR_CLOSE:
            if blink_start is None:
                blink_start = now

        else:
            if blink_start is not None:
                duration = now - blink_start

                # 🔹 SHORT BLINK
                if duration < 0.5:
                     # Check double blink timing
                    if now - last_blink_time < DOUBLE_BLINK_GAP:
                        blink_count += 1
                    else:
                        blink_count = 1

                    last_blink_time = now

                    # DOUBLE BLINK → DOUBLE CLICK
                    if blink_count == 2:
                        pyautogui.doubleClick()
                        blink_count = 0

                    # SINGLE BLINK → SINGLE CLICK
                    elif blink_count == 1:
                        pyautogui.click()

            # 🔹 LONG BLINK → KEYBOARD
                elif duration > 0.6 and now - last_keyboard_time > KEYBOARD_COOLDOWN:
                    print("Opening Virtual Keyboard...")
                    os.system("start osk")
                    last_keyboard_time = now

                blink_start = None

                pyautogui.PAUSE = 0.1  # small delay avoid multiple triggers
        # ===== STATUS DISPLAY =====
        status = "PAUSED" if cursor_paused else "ACTIVE"
        cv2.putText(frame,f"System: {status}",(10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,
                    (0,0,255) if cursor_paused else (0,255,0),2)
        
        

    cv2.imshow("AI Face Controller",frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()