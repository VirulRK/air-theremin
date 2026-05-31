import os
import cv2
import numpy as np
import pygame
import math
import random
import mediapipe as mp
from scipy.io import wavfile
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. GENERATE AUDIO FILES LOCALLY (IF MISSING)
def build_orchestral_sound_files():
    base_frequencies = [261.63, 293.66, 329.63, 392.00, 440.00]
    sample_rate = 44100
    duration = 1.5  # Extended for extra soothing echo ring
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    for i, base_freq in enumerate(base_frequencies):
        filename = f"note{i+1}.wav"
        if not os.path.exists(filename):
            fundamental = np.sin(2 * np.pi * base_freq * t) * 1.0
            cello_body = np.sin(2 * np.pi * (base_freq * 2) * t) * 0.4
            violin_glow = np.sin(2 * np.pi * (base_freq * 3) * t) * 0.25
            warmth = np.sin(2 * np.pi * (base_freq * 0.5) * t) * 0.3
            
            combined = fundamental + cello_body + violin_glow + warmth
            
            envelope = np.zeros_like(t)
            n_samples = len(t)
            attack = int(n_samples * 0.03) # Even sharper attack pluck
            decay = n_samples - attack
            
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[attack:] = np.exp(-2.2 * np.linspace(0, 1, decay))
            
            final_wave = combined * envelope * 0.4
            final_wave = np.clip(final_wave, -1.0, 1.0) * 32767
            wavfile.write(filename, sample_rate, final_wave.astype(np.int16))

build_orchestral_sound_files()

# Ensure AI model exists
model_filename = 'hand_landmarker.task'
if not os.path.exists(model_filename):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_filename)

# 2. INITIALIZE AUDIO
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
notes = [pygame.mixer.Sound(f"note{i+1}.wav") for i in range(5)]
channels = [pygame.mixer.Channel(i) for i in range(5)]

# 3. MEDIAPIPE INITIALIZATION
base_options = python.BaseOptions(model_asset_path=model_filename)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0
last_played_zone = -1

# --- NEW UI/UX ANIMATION DATA STRUCTURES ---
# Track the plucking/vibration state for each horizontal string line
string_vibrations = [0.0] * 4  # 4 string dividers between 5 zones
particles = []                # Holds active particles: [x, y, vx, vy, life, color]

print("\nCinematic UI Canvas Calibrated. Performance Engine Ready.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Smooth trail fade over time
    canvas = cv2.convertScaleAbs(canvas, alpha=0.91, beta=0)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    hud_note = "STANDBY"
    zone_height = h // 5
    
    # --- ANIMATION UPDATE: STRING VIBRATIONS ---
    # Decay string vibration waves smoothly over time using a sine dampening factor
    for idx in range(len(string_vibrations)):
        if string_vibrations[idx] > 0.1:
            string_vibrations[idx] *= 0.88  # Dampen down toward zero string wave velocity
        else:
            string_vibrations[idx] = 0

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            ix = int(hand_landmarks[8].x * w)
            iy = int(hand_landmarks[8].y * h)
            
            current_zone = max(0, min(int(iy // zone_height), 4))
            volume = max(0.1, min((ix / w), 1.0))
            
            # TRIGGER PLUCK EVENT
            if current_zone != last_played_zone:
                channels[current_zone].set_volume(volume)
                channels[current_zone].play(notes[current_zone])
                
                # Ignite String Ripple Wave: Trigger vibration on the boundary line crossed
                crossed_line = current_zone - 1 if iy < (current_zone * zone_height + zone_height // 2) else current_zone
                if 0 <= crossed_line < 4:
                    string_vibrations[crossed_line] = 22.0 * volume # Higher volume = harder physical pluck bounce
                
                # Spawn Particle Burst Sparklers around the fingertips
                for _ in range(12):
                    particles.append([
                        ix, iy,
                        random.uniform(-4, 4),  # Horizontal speed dispersion
                        random.uniform(-5, 2),  # Vertical pop upward acceleration
                        1.0,                    # Lifespan alpha tracking
                        (random.randint(20, 50), random.randint(140, 200), random.randint(230, 255)) # Golden color hues
                    ])
                
                last_played_zone = current_zone
            
            # Draw Golden-Amber brush trail with velocity-based size variation
            if prev_x != 0 and prev_y != 0:
                speed = math.sqrt((ix - prev_x)**2 + (iy - prev_y)**2)
                brush_thickness = max(2, min(int(speed * 0.4), 10)) # Moving faster yields broader glowing sweeps
                cv2.circle(canvas, (ix, iy), brush_thickness // 2, (40, 180, 240), -1)
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (20, 135, 220), brush_thickness)
                
            prev_x, prev_y = ix, iy
            hud_note = f"STRING {5 - current_zone}"
    else:
        prev_x, prev_y = 0, 0
        last_played_zone = -1

    # --- DRAW THE DYNAMIC HUD NET STRINGS ---
    # Draw interactive laser harp divider lines with sine-wave deformation offsets
    for i in range(1, 5):
        y_center = i * zone_height
        vib_amplitude = string_vibrations[i-1]
        
        if vib_amplitude > 0:
            # Generate a series of points to create a curved plucked string look
            points = []
            for x_coord in range(0, w, 15):
                # Standard mathematical wave displacement curve formula
                offset = vib_amplitude * math.sin((x_coord / w) * math.pi) * math.sin(pygame.time.get_ticks() * 0.05)
                points.append((x_coord, int(y_center + offset)))
            
            # Draw smooth interconnected line segments for the string
            for p_idx in range(len(points) - 1):
                cv2.line(frame, points[p_idx], points[p_idx+1], (40, 160, 240), 2, cv2.LINE_AA)
        else:
            # Draw standard sleeping string line
            cv2.line(frame, (0, y_center), (w, y_center), (50, 40, 30), 1, cv2.LINE_AA)

    # --- RENDER AND ANIMATE PARTICLES ---
    for p in particles[:]:
        p[0] += p[2]  # Update X position
        p[1] += p[3]  # Update Y position
        p[3] += 0.15  # Gravity effect gently pulling sparklers downward
        p[4] -= 0.03  # Decay lifetime opacity metric
        
        if p[4] <= 0:
            particles.remove(p)
        else:
            p_radius = max(1, int(4 * p[4]))
            # Drawing a soft anti-aliased glowing orb spark
            cv2.circle(frame, (int(p[0]), int(p[1])), p_radius, p[5], -1, cv2.LINE_AA)

    # Elegant, translucent minimalist HUD Dashboard text panel
    cv2.rectangle(frame, (10, 15), (320, 60), (15, 10, 5), -1)
    cv2.rectangle(frame, (10, 15), (320, 60), (25, 125, 210), 1)
    cv2.putText(frame, f"BAROQUE HARP ENGINE: {hud_note}", (22, 43), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (30, 160, 230), 1, cv2.LINE_AA)
    
    # Merge visual lighting canvas layers with video hardware pipeline output streams
    frame = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

    cv2.imshow('Air-Theremin: Premium Symphony HUD', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
pygame.quit()
cv2.destroyAllWindows()
