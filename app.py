from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob
import gdown
import threading

app = Flask(__name__)
CORS(app)

# ============================================================
# GLOBAL STATE
# ============================================================

predicted_sentence = ""
current_sign = "nothing"

prediction_count = 0
threshold_frames = 15
last_prediction = ""

camera_active = False
camera = None

camera_lock = threading.Lock()
state_lock = threading.Lock()

# ============================================================
# DIRECTORIES
# ============================================================

AUDIO_DIR = os.path.join("static", "audio")
MODEL_DIR = "model"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "asl_model.joblib")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

# ============================================================
# GOOGLE DRIVE MODEL FILES
# ============================================================

model_file_id = "1oZeBgnRUqLYe5IaYG6NIokCEuqz07Ru2"
encoder_file_id = "13oBSsI927KltAI7z0bpz3hgCTrUAQap-"

# ============================================================
# DOWNLOAD MODEL IF MISSING
# ============================================================

if not os.path.exists(model_path):
    print("Downloading ASL model...")

    try:
        gdown.download(
            f"https://drive.google.com/uc?id={model_file_id}",
            model_path,
            quiet=False
        )
    except Exception as e:
        print("Model download failed:", e)


if not os.path.exists(encoder_path):
    print("Downloading label encoder...")

    try:
        gdown.download(
            f"https://drive.google.com/uc?id={encoder_file_id}",
            encoder_path,
            quiet=False
        )
    except Exception as e:
        print("Encoder download failed:", e)


# ============================================================
# LOAD MODEL
# ============================================================

model = None
le = None
model_loaded = False

try:
    if os.path.exists(model_path) and os.path.exists(encoder_path):

        model = joblib.load(model_path)
        le = joblib.load(encoder_path)

        model_loaded = True

        print("========================================")
        print("ASL MODEL LOADED SUCCESSFULLY")
        print("========================================")

    else:
        print("Model files not found.")

except Exception as e:

    model_loaded = False

    print("========================================")
    print("MODEL LOADING ERROR")
    print("========================================")
    print(e)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)


# ============================================================
# THUMB UP DETECTION
# ============================================================

CUSTOM_GESTURES = {
    "okay": "OKAY",
    "no": "NO",
    "hello": "HELLO",
    "hi": "HI",
    "perfect": "PERFECT",
    "i love you": "I LOVE YOU",
    "good luck": "GOOD LUCK",
    "rock": "ROCK",
    "one": "ONE",
    "stop": "STOP",
    "wait": "WAIT",
    "five": "FIVE",
    "fist": "FIST",
    "small": "SMALL",
    "what": "WHAT",
    "down": "DOWN",
    "up": "UP",
    "right": "RIGHT",
    "left": "LEFT",
    "please": "PLEASE",
    "thank you": "THANK YOU",
    "applause": "APPLAUSE",
    "deal": "DEAL",
    "strong": "STRONG",
    "write": "WRITE",
    "call me": "CALL ME",
}


def _finger_states(hand_landmarks):
    lm = hand_landmarks.landmark

    return {
        "index": lm[8].y < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring": lm[16].y < lm[14].y,
        "pinky": lm[20].y < lm[18].y,
    }


def _thumb_is_extended(hand_landmarks):
    lm = hand_landmarks.landmark
    return abs(lm[4].x - lm[0].x) > abs(lm[3].x - lm[0].x)


def detect_custom_gesture(hand_landmarks):
    """Return a custom gesture label when the landmark pose matches one."""
    lm = hand_landmarks.landmark
    fingers = _finger_states(hand_landmarks)
    extended_count = sum(fingers.values())
    thumb_up = lm[4].y < lm[3].y < lm[2].y
    thumb_down = lm[4].y > lm[3].y > lm[2].y
    thumb_extended = _thumb_is_extended(hand_landmarks)
    index_middle_touching = abs(lm[8].x - lm[12].x) < 0.06 and abs(lm[8].y - lm[12].y) < 0.12
    thumb_index_touching = abs(lm[4].x - lm[8].x) < 0.08 and abs(lm[4].y - lm[8].y) < 0.12

    if thumb_up and extended_count == 0:
        return "okay"
    if thumb_down and extended_count == 0:
        return "no"
    if fingers == {"index": True, "middle": True, "ring": False, "pinky": False}:
        return "hello"
    if fingers == {"index": True, "middle": False, "ring": False, "pinky": True} and thumb_extended:
        return "i love you"
    if fingers == {"index": True, "middle": False, "ring": False, "pinky": True}:
        return "rock"
    if fingers == {"index": True, "middle": True, "ring": False, "pinky": False} and index_middle_touching:
        return "good luck"
    if fingers == {"index": False, "middle": False, "ring": False, "pinky": True} and thumb_extended:
        return "call me"
    if thumb_index_touching and extended_count == 0:
        return "small"
    if fingers == {"index": True, "middle": False, "ring": False, "pinky": False}:
        if lm[8].x > lm[0].x + 0.18:
            return "right"
        if lm[8].x < lm[0].x - 0.18:
            return "left"
        return "one"
    if thumb_extended and extended_count == 0:
        return "strong"
    if extended_count == 0:
        return "fist"
    if thumb_index_touching and extended_count >= 2:
        return "perfect"
    if extended_count == 4:
        return "five"
    return None


def is_thumb_up(hand_landmarks):
    """
    Detect a simple thumb-up gesture.

    Thumb:
        tip = 4
        IP  = 3
        MCP = 2

    Fingers:
        Index  = 8 / 6
        Middle = 12 / 10
        Ring   = 16 / 14
        Pinky  = 20 / 18
    """

    lm = hand_landmarks.landmark

    # Thumb must be pointing upward
    thumb_up = (
        lm[4].y < lm[3].y and
        lm[3].y < lm[2].y
    )

    # Other fingers should be folded
    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y

    fingers_folded = (
        index_folded and
        middle_folded and
        ring_folded and
        pinky_folded
    )

    return thumb_up and fingers_folded


# ============================================================
# CAMERA CLASS
# ============================================================

class VideoCamera:

    def __init__(self):

        print("Opening camera...")

        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.video.isOpened():

            # Fallback
            self.video.release()

            self.video = cv2.VideoCapture(0)

        if not self.video.isOpened():

            raise RuntimeError(
                "Could not open camera. Camera may be busy or unavailable."
            )

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Try to reduce buffering
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Camera opened successfully.")


    def get_frame(self):

        global predicted_sentence
        global current_sign
        global prediction_count
        global last_prediction

        ret, frame = self.video.read()

        if not ret:

            print("Failed to read camera frame.")

            return None

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        detected_label = "nothing"

        if result.multi_hand_landmarks:

            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            custom_label = detect_custom_gesture(hand_landmarks)

            if custom_label:

                detected_label = custom_label

            # =================================================
            # NORMAL ASL MODEL
            # =================================================

            elif model_loaded:

                features = []

                for lm in hand_landmarks.landmark:

                    features.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                x_input = np.array(
                    features,
                    dtype=np.float32
                ).reshape(1, -1)

                try:

                    y_pred = model.predict(x_input)

                    detected_label = le.inverse_transform(
                        y_pred
                    )[0]

                except Exception as e:

                    print("Prediction error:", e)

                    detected_label = "nothing"


        # ====================================================
        # CURRENT SIGN
        # ====================================================

        current_sign = detected_label


        # ====================================================
        # STABILITY
        # ====================================================

        if detected_label == last_prediction:

            prediction_count += 1

        else:

            prediction_count = 0
            last_prediction = detected_label


        # ====================================================
        # ADD SIGN TO SENTENCE
        # ====================================================

        if prediction_count >= threshold_frames:

            if detected_label == "space":

                predicted_sentence += " "

            elif detected_label == "del":

                predicted_sentence = predicted_sentence[:-1]

            elif detected_label in CUSTOM_GESTURES:

                if predicted_sentence and not predicted_sentence.endswith(" "):
                    predicted_sentence += " "

                predicted_sentence += detected_label

            elif detected_label != "nothing":

                predicted_sentence += detected_label

            # Reset
            prediction_count = 0


        # ====================================================
        # CAMERA OVERLAY
        # ====================================================

        # Top black panel
        cv2.rectangle(
            frame,
            (10, 10),
            (630, 105),
            (0, 0, 0),
            -1
        )

        # Sign
        cv2.putText(
            frame,
            f"Sign: {current_sign}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        # Sentence
        sentence_display = predicted_sentence

        if len(sentence_display) > 45:
            sentence_display = sentence_display[-45:]

        cv2.putText(
            frame,
            f"Sentence: {sentence_display}",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # JPEG
        # ====================================================

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:

            return None

        return buffer.tobytes()


    def release(self):

        if self.video is not None:

            self.video.release()

            print("Camera released.")


# ============================================================
# VIDEO GENERATOR
# ============================================================

def gen_frames():

    global camera_active

    while camera_active:

        try:

            with camera_lock:

                current_camera = camera

                if current_camera is None:

                    break

                frame = current_camera.get_frame()


            if frame is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n"
                    b"Pragma: no-cache\r\n\r\n"
                    + frame +
                    b"\r\n"
                )

            else:

                time.sleep(0.05)

        except Exception as e:

            print("Video stream error:", e)

            break

        time.sleep(1 / 30)


# ============================================================
# AUDIO
# ============================================================

def cleanup_old_audio_files():

    try:

        cutoff_time = datetime.now() - timedelta(hours=1)

        audio_files = glob.glob(
            os.path.join(AUDIO_DIR, "*.mp3")
        )

        for file_path in audio_files:

            file_time = datetime.fromtimestamp(
                os.path.getctime(file_path)
            )

            if file_time < cutoff_time:

                try:
                    os.remove(file_path)
                except:
                    pass

    except Exception as e:

        print("Audio cleanup error:", e)


def generate_audio_file(text):

    try:

        if not text or not text.strip():

            return None

        cleanup_old_audio_files()

        filename = (
            f"speech_{uuid.uuid4().hex[:8]}.mp3"
        )

        filepath = os.path.join(
            AUDIO_DIR,
            filename
        )

        tts = gTTS(
            text=text,
            lang="en",
            slow=False
        )

        tts.save(filepath)

        return filename

    except Exception as e:

        print("TTS error:", e)

        return None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# START CAMERA
# ============================================================

@app.route("/start_camera", methods=["POST"])
def start_camera():

    global camera_active
    global camera

    with camera_lock:

        if camera_active and camera is not None:

            return jsonify({
                "status": "success",
                "message": "Camera already running"
            })

        try:

            camera = VideoCamera()

            # Test frame BEFORE telling frontend camera started
            test_frame = camera.get_frame()

            if test_frame is None:

                camera.release()
                camera = None

                return jsonify({
                    "status": "error",
                    "message": "Camera opened but frame could not be read."
                }), 500

            camera_active = True

            print("Camera STARTED.")

            return jsonify({
                "status": "success",
                "message": "Camera started"
            })

        except Exception as e:

            camera = None
            camera_active = False

            print("Camera start error:", e)

            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    if not camera_active:

        return "Camera is not active", 400

    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


# ============================================================
# STOP CAMERA
# ============================================================

@app.route("/stop_camera", methods=["POST"])
def stop_camera():

    global camera_active
    global camera

    with camera_lock:

        camera_active = False

        if camera is not None:

            camera.release()

            camera = None

    print("Camera STOPPED.")

    return jsonify({
        "status": "success",
        "message": "Camera stopped"
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/get_sentence")
def get_sentence():

    return jsonify({

        "sentence": predicted_sentence,

        "current_sign": current_sign,

        "prediction_count": prediction_count,

        "threshold_frames": threshold_frames,

        "camera_active": camera_active

    })


# ============================================================
# CHATBOT CONFIRMATION
# ============================================================

@app.route("/chatbot_response", methods=["POST"])
def chatbot_response():

    payload = request.get_json(silent=True) or {}
    sign = str(payload.get("sign", "nothing")).strip().lower()
    confirmed = payload.get("confirmed") is True

    routine_messages = {
        "hello": "Hello, how are you?",
        "hi": "Hello, how are you?",
        "good morning": "Good morning, how are you?",
        "goodbye": "Goodbye, see you later.",
        "water": "I need water.",
        "food": "I need food.",
        "eat": "I want to eat.",
        "tea": "I would like tea.",
        "hungry": "I am hungry.",
        "thirsty": "I am thirsty.",
        "bathroom": "I need to use the bathroom.",
        "sleep": "I want to sleep.",
        "wake up": "Please wake me up.",
        "medicine": "I need my medicine.",
        "pain": "I have pain here.",
        "sick": "I am sick.",
        "tired": "I am tired.",
        "cold": "I feel cold.",
        "hot": "I feel hot.",
        "help": "Please help me.",
        "emergency": "This is an emergency. I need help.",
        "doctor": "I need a doctor.",
        "police": "Please call the police.",
        "fire": "There is a fire. Please help.",
        "i am okay": "I am okay.",
        "i need help": "I need help.",
    }

    next_steps = {
        "help": "If you need immediate assistance, show HELP again and contact someone nearby.",
        "water": "You can confirm WATER, then continue with another sign or press Speak Sentence.",
        "bathroom": "You can add more signs, then press Speak Sentence to communicate the request.",
        "pain": "Show the body location next, then press Speak Sentence to explain the request.",
        "stop": "The next step is to stop the current action and show another sign when ready.",
        "thank you": "You can continue with another sign or press Speak Sentence.",
    }

    if confirmed:
        return jsonify({
            "status": "confirmed",
            "message": f"I understand you are trying to say {sign.upper()}.",
            "suggested_message": routine_messages.get(sign, ""),
            "next_step": next_steps.get(
                sign,
                "Continue showing the next sign, or press Speak Sentence to share this message."
            )
        })

    return jsonify({
        "status": "retry",
        "message": "No problem. Hold the sign steady and try again.",
        "next_step": "I will wait for your next clear sign."
    })


# ============================================================
# CLEAR
# ============================================================

@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():

    global predicted_sentence
    global prediction_count
    global last_prediction

    predicted_sentence = ""

    prediction_count = 0
    last_prediction = ""

    return jsonify({
        "status": "success",
        "message": "Sentence cleared"
    })


# ============================================================
# SPEAK
# ============================================================

@app.route("/speak_sentence", methods=["POST"])
def speak_sentence():

    if not predicted_sentence.strip():

        return jsonify({

            "status": "error",

            "message": "No sentence to speak"

        }), 400


    audio_filename = generate_audio_file(
        predicted_sentence
    )


    if audio_filename:

        return jsonify({

            "status": "success",

            "sentence": predicted_sentence,

            "audio_url":
                f"/static/audio/{audio_filename}",

            "message":
                "Audio generated successfully"

        })


    return jsonify({

        "status": "error",

        "message": "Failed to generate audio"

    }), 500


# ============================================================
# AUDIO
# ============================================================

@app.route("/static/audio/<filename>")
def serve_audio(filename):

    return send_from_directory(
        AUDIO_DIR,
        filename
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("")
    print("==========================================")
    print("     ASL SIGN LANGUAGE PROJECT")
    print("==========================================")
    print("")
    print("Open: http://127.0.0.1:5000")
    print("")

    # IMPORTANT:
    # use_reloader=False prevents Windows camera conflicts
    app.run(
        debug=True,
        use_reloader=False,
        threaded=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )