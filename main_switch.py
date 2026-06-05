import subprocess
import time
import keyboard # Install: pip install keyboard
import os

# --- Path Settings ---
#
HAND_SCRIPT = os.path.join("hand_module", "control.py") 
HEAD_SCRIPT = os.path.join("head_module", "head_cursorr.py")

current_process = None
current_mode = None 

def stop_script():
    global current_process
    if current_process:
        print("Stopping current script...")
        
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(current_process.pid)])
        current_process = None

def start_script(script_path):
    global current_process
    stop_script()
    
    if os.path.exists(script_path):
        print(f"Starting: {script_path}")
        
        script_dir = os.path.dirname(script_path)
        current_process = subprocess.Popen(['python', os.path.basename(script_path)], cwd=script_dir)
    else:
        print(f"Error: {script_path} ! check the path.")

print("--- System Ready ---")
print("Press '1' for HAND Mode (Folder: hand)")
print("Press '2' for HEAD Mode (Folder: head)")
print("Press 'Q' to Quit")

while True:
    try:
        # Mode 1: Hand Control
        if keyboard.is_pressed('1'):
            if current_mode != 'hand':
                start_script(HAND_SCRIPT)
                current_mode = 'hand'
            time.sleep(0.5)

        # Mode 2: Head Control
        if keyboard.is_pressed('2'):
            if current_mode != 'head':
                start_script(HEAD_SCRIPT)
                current_mode = 'head'
            time.sleep(0.5)

        # Quit
        if keyboard.is_pressed('q'):
            stop_script()
            break

        time.sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")
        break

print("System Exited.")