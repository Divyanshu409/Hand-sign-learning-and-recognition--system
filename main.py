import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import base64

app = Flask(__name__)

# ── Labels ────────────────────────────────────────────────────────────────────
with open("Model/labels.txt", "r") as f:
    lines = f.read().splitlines()

labels = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split(" ", 1)
    labels.append(parts[1] if len(parts) == 2 else parts[0])

NUM_CLASSES = len(labels)
print(f"[OK] Loaded {NUM_CLASSES} labels: {labels}")

# ── TFLite model ──────────────────────────────────────────────────────────────
TFLITE_PATH = "Model/model.tflite"

interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SHAPE = input_details[0]['shape']
INPUT_DTYPE = input_details[0]['dtype']
IMG_SIZE    = INPUT_SHAPE[1]

if INPUT_DTYPE == np.uint8:
    NORM_MODE = "uint8"
else:
    quant = input_details[0].get('quantization', (0.0, 0))
    NORM_MODE = "mobilenet"

print(f"[OK] TFLite model loaded | input shape={INPUT_SHAPE} | dtype={INPUT_DTYPE} | norm={NORM_MODE}")

# Warm up
dummy = np.zeros(INPUT_SHAPE, dtype=INPUT_DTYPE)
interpreter.set_tensor(input_details[0]['index'], dummy)
interpreter.invoke()
_ = interpreter.get_tensor(output_details[0]['index'])
print("[OK] Model warmed up.")

# ── Per-session state (single user; extend to sessions for multi-user) ────────
ema_pred         = None
recent_indices   = []
hold_counter     = 0
hold_label       = ""
no_hand_frames   = 0
current_sign_label      = ""
current_sign_confidence = 0.0
sentence   = ""
last_added = ""

# ── Thresholds ────────────────────────────────────────────────────────────────
EMA_ALPHA        = 0.35
CONF_THRESHOLD   = 0.55
STABILITY_THRESH = 0.70
ENTROPY_THRESH   = 3.2
AMBIGUITY_MARGIN = 0.15
HOLD_FRAMES      = 10
NO_HAND_GRACE    = 8
STABILITY_WINDOW = 12

# ── Helpers ───────────────────────────────────────────────────────────────────
def tflite_predict(img_array: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    if output.dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)

def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if NORM_MODE == "uint8":
        arr = rgb.astype(np.uint8)
    elif NORM_MODE == "rescale":
        arr = rgb.astype(np.float32) / 255.0
    else:
        arr = (rgb.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def compute_entropy(pred: np.ndarray) -> float:
    p = np.clip(pred, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

def is_ambiguous(pred: np.ndarray, margin: float = AMBIGUITY_MARGIN) -> bool:
    sorted_p = np.sort(pred)[::-1]
    return (sorted_p[0] - sorted_p[1]) < margin

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/current_sign')
def current_sign():
    return jsonify({
        "label":      current_sign_label,
        "confidence": round(current_sign_confidence * 100, 1)
    })

@app.route('/sentence')
def get_sentence():
    return jsonify({"sentence": sentence})

@app.route('/clear_sentence')
def clear_sentence():
    global sentence, last_added
    sentence   = ""
    last_added = ""
    return jsonify({"status": "cleared"})

@app.route('/gestures/<category>')
def gestures(category):
    if category.lower() == "alphabets":
        return jsonify({
            "data": [
                {
                    "name":    chr(i),
                    "meaning": f"Letter {chr(i)} sign",
                    "image":   f"/static/images/{chr(i)}.png"
                }
                for i in range(65, 91)
            ]
        })
    elif category.lower() == "basic":
        return jsonify({
            "data": [
                {"name": "0", "meaning": "digit number 0", "image": "/static/images/bye.jpg"},
                {"name": "1", "meaning": "digit number 1", "image": "/static/images/dislike.jpg"},
                {"name": "2", "meaning": "digit number 2", "image": "/static/images/hello.jpg"},
                {"name": "3", "meaning": "digit number 3", "image": "/static/images/help_me.jpg"},
                {"name": "4", "meaning": "digit number 4", "image": "/static/images/like.jpg"},
                {"name": "5", "meaning": "digit number 5", "image": "/static/images/no.jpg"},
                {"name": "6", "meaning": "digit number 6", "image": "/static/images/please.jpg"},
                {"name": "7", "meaning": "digit number 7", "image": "/static/images/thank_you.jpg"},
                {"name": "8", "meaning": "digit number 8", "image": "/static/images/thisismyproject.jpg"},
                {"name": "9", "meaning": "digit number 9", "image": "/static/images/yes.jpg"},
            ]
        })
    return jsonify({"data": []})

# ── /predict  (replaces /video_feed) ─────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a base64-encoded JPEG crop of the hand region from the browser,
    runs TFLite inference, updates session state, and returns JSON.

    Request body (JSON):
        { "image": "<base64 JPEG string>", "has_hand": true/false }

    Response (JSON):
        {
          "label": "A",
          "confidence": 87.3,
          "stability": 75,
          "hold_counter": 3,
          "hold_total": 10,
          "sentence": "HELLO "
        }
    """
    global ema_pred, recent_indices, hold_counter, hold_label
    global no_hand_frames, current_sign_label, current_sign_confidence
    global sentence, last_added

    data = request.get_json(force=True)
    has_hand = data.get('has_hand', False)

    if not has_hand:
        no_hand_frames += 1
        if no_hand_frames >= NO_HAND_GRACE:
            current_sign_label      = ""
            current_sign_confidence = 0.0
            ema_pred        = None
            recent_indices  = []
            hold_counter    = 0
            hold_label      = ""
            no_hand_frames  = 0
        return jsonify({
            "label": "", "confidence": 0,
            "stability": 0, "hold_counter": 0,
            "hold_total": HOLD_FRAMES, "sentence": sentence
        })

    no_hand_frames = 0

    # Decode base64 image
    try:
        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]
        img_bytes = base64.b64decode(img_b64)
        img_np    = np.frombuffer(img_bytes, dtype=np.uint8)
        crop_bgr  = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Could not decode image")
        # NOTE: Do NOT flip here. The browser crops from the raw (unmirrored)
        # video frame and sends that directly. The old server-side pipeline
        # flipped the full frame before cropping; doing it again on the crop
        # would double-mirror the hand and break recognition.
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Inference
    img_input  = preprocess(crop_bgr)
    prediction = tflite_predict(img_input)

    # EMA smoothing
    if ema_pred is None:
        ema_pred = prediction.copy()
    else:
        ema_pred = EMA_ALPHA * prediction + (1.0 - EMA_ALPHA) * ema_pred

    smooth_idx = int(np.argmax(ema_pred))
    confidence = float(ema_pred[smooth_idx])

    # Stability window
    recent_indices.append(smooth_idx)
    if len(recent_indices) > STABILITY_WINDOW:
        recent_indices.pop(0)
    stability = recent_indices.count(smooth_idx) / len(recent_indices)

    entropy   = compute_entropy(ema_pred)
    label     = labels[smooth_idx] if smooth_idx < len(labels) else "Unknown"

    current_sign_label      = label
    current_sign_confidence = confidence

    ambiguous = is_ambiguous(ema_pred)

    gates_ok = (
        confidence > CONF_THRESHOLD
        and stability  > STABILITY_THRESH
        and entropy    < ENTROPY_THRESH
        and not ambiguous
    )

    if gates_ok:
        if label == hold_label:
            hold_counter += 1
        else:
            hold_label   = label
            hold_counter = 1

        if hold_counter >= HOLD_FRAMES and label != last_added:
            sentence  += " " if label == "space" else label
            last_added = label
            hold_counter = 0
    else:
        hold_counter = 0

    return jsonify({
        "label":       label,
        "confidence":  round(confidence * 100, 1),
        "stability":   round(stability * 100),
        "hold_counter": hold_counter,
        "hold_total":   HOLD_FRAMES,
        "sentence":    sentence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)