pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
=============================
python - << "PY"
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
========================================
pip install nvidia-tensorrt==8.6.1
++++++++++++++++++++++++++++++++++++++++
python -c "import tensorrt as trt; print(trt.__version__)"
================================




from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, jsonify
import cv2
from matplotlib import pyplot as plt
import traceback
import pandas as pd
from ultralytics import YOLO
from collections import Counter
import datetime
import csv
import io
import threading
import os
import requests
from pathlib import Path
import time
from bing_image_downloader import downloader
import queue
from flask_cors import CORS
import numpy as np
import json

# ========= GPU enforcement =========
import torch
torch.backends.cudnn.benchmark = True  

# HARD REQUIREMENT: GPU must be available
if not torch.cuda.is_available():
    raise RuntimeError(
        "\n FATAL: No NVIDIA GPU detected. This application is GPU-only.\n"
        "Ensure the following:\n"
        "  • NVIDIA driver installed (nvidia-smi works)\n"
        "  • PyTorch CUDA 11.8 (cu118) installed\n"
        "  • TensorRT runtime installed\n"
        "  • Engine file built for this machine/GPU\n"
    )

# ( pin to GPU:0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

print("====================================================")
print(" GPU:", torch.cuda.get_device_name(0))
print(" CUDA Runtime:", torch.version.cuda)
print("====================================================")

# Try TensorRT import (required when using .engine)
try:
    import tensorrt as trt
    print(" TensorRT:", trt.__version__)
except Exception as e:
    raise RuntimeError(
        " TensorRT Python package not found. Install TensorRT 8.x for CUDA 11.x.\n"
        "Engine inference requires TensorRT runtime."
    ) from e

# ========= Project constants =========
ENGINE_PATH = "yolov8n.engine"   # TRT engine file
PT_WEIGHTS = "yolov8n.pt"        # for training (GPU)
IMG_SIZE = 640
DEVICE = 0                       # GPU:0 only

# ========= Flask and globals =========
app = Flask(__name__)
CORS(app)

all_detections = []
current_counts = Counter()
is_detecting = False
image_save_interval = 5
ocr_image_save_interval = 10
is_training = False
last_detection_time = 0
detection_mode = "OBJECT"  # OBJECT | OCR | BARCODE
ocr_detected_chars = ""
all_detected_chars = []  # [("char", timestamp)]

ocr_queue = queue.Queue()
frame_queue = queue.Queue()

barcode_detected = ""
all_detected_barcodes = []
seen_barcodes = set()
barcode_product_data = []

# OBJECT DETECTION CONTROL VARIABLE
object_detection_enabled = True  # default enabled

# WEIGHT VARIABLES
WEIGHT_SERVICE_URL = "http://localhost:5004"
weight_monitoring_enabled = False
all_detected_weights = []
calibration_file = "on-demand-features/weights/calibration.json"

# Paths
LABELS_FOLDER = 'train/labels'
IMAGE_FOLDER = 'train/images'
COUNTER_FILE = 'label_counter.txt'
REPORTS_FOLDER = 'static/reports'

os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# label/img counters
if not os.path.isfile(COUNTER_FILE):
    with open(COUNTER_FILE, 'w') as counter_file:
        counter_file.write('1')

with open(COUNTER_FILE, 'r') as counter_file:
    label_counter = int(counter_file.read().strip())
    img_counter = label_counter

LABEL_STUDIO_API_URL = 'http://localhost:8080/api/projects/{project_id}/import'
LABEL_STUDIO_API_KEY = '7f6664a6e9a473ea148537b8d367e55d1793b48b'
PROJECT_ID = '3'

label_map = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5,
    "train": 6, "truck": 7, "boat": 8, "traffic light": 9, "fire hydrant": 10,
    "stop sign": 11, "parking meter": 12, "bench": 13, "bird": 14, "cat": 15,
    "dog": 16, "horse": 17, "sheep": 18, "cow": 19, "elephant": 20, "bear": 21,
    "zebra": 22, "giraffe": 23, "backpack": 24, "umbrella": 25, "handbag": 26,
    "tie": 27, "suitcase": 28, "frisbee": 29, "skis": 30, "snowboard": 31,
    "sports ball": 32, "kite": 33, "baseball bat": 34, "baseball glove": 35,
    "skateboard": 36, "surfboard": 37, "tennis racket": 38, "bottle": 39,
    "wine glass": 40, "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
    "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59, "dining table": 60,
    "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64, "remote": 65, "keyboard": 66,
    "cell phone": 67, "microwave": 68, "oven": 69, "toaster": 70, "sink": 71,
    "refrigerator": 72, "book": 73, "clock": 74, "vase": 75, "scissors": 76,
    "teddy bear": 77, "hair drier": 78, "toothbrush": 79
}

# ========= Database I/O  =========
from database.database import (
    save_image_detection,
    save_text_detection,
    save_barcode_scan,
    get_image_detections,
    get_text_detections,
    get_barcode_scans
)

# ========= Load TensorRT engine (GPU-only) =========
print("\n Loading TensorRT engine on GPU...")
if not os.path.exists(ENGINE_PATH):
    raise FileNotFoundError(f" Engine file not found: {ENGINE_PATH}")
model = YOLO(ENGINE_PATH)

backend = getattr(model, 'predictor', None)
print(" CUDA available:", torch.cuda.is_available())
print(" GPU Device:", torch.cuda.get_device_name(0))
print(" Engine Loaded:", ENGINE_PATH)
print(" Predictor Backend:",
      backend.__class__.__name__ if backend else " None (TensorRT NOT active)")
print(" Ultralytics Device:", getattr(model, 'device', 'unknown'))
print("====================================================\n")


# =========================
# Flask routes
# ========================
"""
Function  : index()
Summary   : Render the main index page.
Params    : none
Return    : Rendered HTML template for the main page ('index.html').
"""

@app.route('/')
def index():
    return render_template('index.html')
    
# OBJECT DETECTION TOGGLE ENDPOINT
@app.route('/toggle_object_detection', methods=['POST'])
def toggle_object_detection():
    global object_detection_enabled
    try:
        enable = request.form.get('enable', 'false').lower() == 'true'
        object_detection_enabled = enable
        status_msg = "Object detection enabled" if enable else "Object detection disabled"
        return jsonify({'status': 'success', 'message': status_msg, 'object_enabled': object_detection_enabled})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error toggling object detection: {str(e)}'})

# GET OBJECT STATUS ENDPOINT  
@app.route('/get_object_status')
def get_object_status():
    return jsonify({
        'object_enabled': object_detection_enabled,
        'weight_enabled': weight_monitoring_enabled,
        'service_type': 'OBJECT',
        'port': 5001,
        'total_detections': len(all_detections),
        'latest_detection': dict(all_detections[0][0]) if all_detections else None,
        'is_training': is_training
    })
    
# GET OBJECT STATUS ENDPOINT  
@app.route('/toggle_weight_monitoring', methods=['POST'])
def toggle_weight_monitoring():
    global weight_monitoring_enabled
    try:
        enable = request.form.get('enable', 'false').lower() == 'true'
        if enable and not weight_monitoring_enabled:
            try:
                test_response = requests.get(f"{WEIGHT_SERVICE_URL}/", timeout=3)
                if test_response.status_code != 200:
                    return jsonify({'status': 'error', 'message': 'Weight service not available'})
            except:
                return jsonify({'status': 'error', 'message': 'Weight service not running on port 5004'})
            # calibration check
            skip_calibration = False
            try:
                with open(calibration_file, 'r') as file:
                    calibration_data = json.load(file)
                    if 'offset' in calibration_data and 'scale' in calibration_data:
                        skip_calibration = True
                        print("Calibration already done, skipping configuration")
                    known_weight = calibration_data.get('known_weight', 0)
            except:
                known_weight = 0

            if not skip_calibration:
                config_response = requests.post(f"{WEIGHT_SERVICE_URL}/configure_weight_setup",
                                                json={"known_weight": known_weight})
                if config_response.status_code != 200:
                    return jsonify({'status': 'error', 'message': 'Failed to configure weight service'})

            response = requests.post(f"{WEIGHT_SERVICE_URL}/start_weight_measurement",
                                     json={'interval': image_save_interval}, timeout=5)
            if response.status_code == 200:
                weight_monitoring_enabled = True
                updateCalibrationFile(weight_monitoring_enabled)
                return jsonify({'status': 'success', 'message': 'Object weight monitoring enabled'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to start weight monitoring'})

        elif not enable and weight_monitoring_enabled:
            try:
                requests.post(f"{WEIGHT_SERVICE_URL}/stop_weight_measurement", timeout=5)
            except:
                pass
            weight_monitoring_enabled = False
            updateCalibrationFile(weight_monitoring_enabled)
            return jsonify({'status': 'success', 'message': 'Object weight monitoring disabled'})

        return jsonify({'status': 'info', 'message': f'Object weight monitoring already {"enabled" if enable else "disabled"}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error toggling object weight monitoring: {str(e)}'})

def updateCalibrationFile(flag: bool):
    try:
        with open(calibration_file, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    existing_data.update({"weight_monitoring_enabled": flag})
    with open(calibration_file, "w") as f:
        json.dump(existing_data, f)

@app.route('/check_weight_service')
def check_weight_service():
    try:
        response = requests.get(f"{WEIGHT_SERVICE_URL}/", timeout=3)
        return jsonify({'available': True, 'service': response.json()})
    except:
        return jsonify({'available': False, 'message': 'Weight service not running'})

@app.route('/get_weight_status')
def get_weight_status():
    return jsonify({
        'weight_enabled': weight_monitoring_enabled,
        'service_url': WEIGHT_SERVICE_URL,
        'service_type': 'OBJECT',
        'latest_weight': all_detected_weights[-1] if all_detected_weights else None,
        'total_measurements': len(all_detected_weights)
    })

# Serve saved training images
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/downloads/<path:filename>')
def serve_downloads(filename):
    return send_from_directory("downloads", filename)

@app.route('/label_studio_webhook', methods=['POST'])
def handle_label_studio_webhook():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid data"}), 400

    annotation = data.get('annotation', {}).get('result', [])
    if not annotation:
        return jsonify({"error": "No annotation data found"}), 400

    label_counter_local = 1
    for result in annotation:
        original_width = result['original_width']
        original_height = result['original_height']

        bbox_value = result['value']
        x = bbox_value['x']
        y = bbox_value['y']
        width = bbox_value['width']
        height = bbox_value['height']

        x_center = (x + width / 2) / original_width
        y_center = (y + height / 2) / original_height
        width_normalized = width / original_width
        height_normalized = height / original_height

        label_name = bbox_value['rectanglelabels'][0]
        class_id = label_map.get(label_name, -1)
        if class_id == -1:
            return jsonify({"error": f"Label '{label_name}' not found in label_map"}), 400

        yolo_label = f"{class_id} {x_center} {y_center} {width_normalized} {height_normalized}\n"
        label_filename = os.path.join('train/labels', f"{label_counter_local}.txt")
        with open(label_filename, 'w') as label_file:
            label_file.write(yolo_label)
        label_counter_local += 1

    print("Received data from Label Studio:", data)
    return jsonify({"status": "success"}), 200

@app.route('/search_and_download', methods=['POST'])
def search_and_download():
    global label_counter
    search_term = request.form.get('search_term')
    num_images = int(request.form.get('num_images', 10))
    if not search_term:
        return jsonify({'status': 'error', 'message': 'No search term provided'}), 400

    try:
        temp_dir = "downloads"
        os.makedirs(temp_dir, exist_ok=True)

        downloader.download(search_term, limit=num_images, output_dir=temp_dir,
                            adult_filter_off=True, force_replace=False, timeout=60)

        downloaded_images = []
        for i, filename in enumerate(os.listdir(temp_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                old_path = os.path.join(temp_dir, filename)
                new_name = f"{label_counter + i}.jpg"
                new_path = os.path.join(IMAGE_FOLDER, new_name)
                os.rename(old_path, new_path)
                downloaded_images.append(new_name)

        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        label_counter += len(downloaded_images)
        with open(COUNTER_FILE, 'w') as counter_file:
            counter_file.write(str(label_counter))

        return jsonify({
            'status': 'success',
            'message': f'Downloaded {len(downloaded_images)} images for "{search_term}"',
            'images': downloaded_images
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error downloading images: {str(e)}'}), 500

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global is_detecting, last_detection_time, label_counter
    global ocr_detected_chars, all_detected_chars, barcode_detected, all_detected_barcodes
    global object_detection_enabled, weight_monitoring_enabled, all_detected_weights

    if not object_detection_enabled:
        return jsonify({'status': 'disabled', 'message': 'Object detection is disabled', 'mode': 'OBJECT'})

    current_time = time.time()
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received'}), 400

    file = request.files['image']
    image_bytes = file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    target_width = request.form.get('width')
    target_height = request.form.get('height')
    if target_width and target_height:
        frame = cv2.resize(frame, (int(target_width), int(target_height)))
    else:
        h, w = frame.shape[:2]
        target_width, target_height = w, h

    if is_detecting:
        if current_time - last_detection_time >= image_save_interval:
            boxes = detect_objects(frame)
            boxes = segment_objects(frame, boxes, int(target_height), int(target_width))

            image_filename = f'{label_counter}.jpg'
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            cv2.imwrite(image_path, frame)
            label_counter += 1
            with open(COUNTER_FILE, 'w') as counter_file:
                counter_file.write(str(label_counter))
            last_detection_time = current_time

            return jsonify({"status": "frame processed", "boxes": boxes, "mode": "OBJECT"})

    return jsonify({'status': 'frame processed'})

def detect_objects(frame):
    """Run YOLO inference (TensorRT engine) on GPU:0 only."""
    global all_detections, current_counts, weight_monitoring_enabled, all_detected_weights
    current_weight = None

    # === GPU ONLY inference ===
    results = model.predict(frame, device=DEVICE, imgsz=IMG_SIZE, verbose=False)

    current_counts.clear()
    detected_objects = {}
    bounding_boxes = []

    for r in results:
        for box in r.boxes:
            c = int(box.cls)
            class_name = model.names[c] if hasattr(model, 'names') else str(c)
            current_counts[class_name] += 1
            detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            bounding_boxes.append({
                "label": class_name,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "confidence": confidence,
                "mode": "OBJECT"
            })

    timestamp = datetime.datetime.utcnow()

    if detected_objects:
        save_image_detection({"objects": detected_objects, "timestamp": timestamp})

    if not weight_monitoring_enabled:
        try:
            with open(calibration_file, 'r') as file:
                weight_monitoring_enabled = json.load(file).get('weight_monitoring_enabled', False)
        except FileNotFoundError:
            weight_monitoring_enabled = False

    if weight_monitoring_enabled and detected_objects:
        try:
            weight_response = requests.get(f"{WEIGHT_SERVICE_URL}/get_current_weight", timeout=3)
            if weight_response.status_code == 200:
                weight_data = weight_response.json()
                current_weight = weight_data.get('weight', 0.0)
                weight_timestamp = weight_data.get('timestamp', datetime.datetime.now().isoformat())

                from database.database import save_weight_measurement
                object_weight_data = {
                    "weight": current_weight,
                    "timestamp": datetime.datetime.now(),
                    "unit": "grams",
                    "detected_objects": detected_objects,
                    "object_count": sum(detected_objects.values()),
                    "detection_type": "object"
                }
                save_weight_measurement(object_weight_data)
                all_detected_weights.append((current_weight, weight_timestamp))
                all_detected_weights = all_detected_weights[-100:]
            else:
                print("Weight service unavailable")
        except Exception as e:
            print(f"Object weight service error: {e}")

    all_detections.insert(0, (dict(current_counts), timestamp.strftime("%H:%M:%S"), current_weight))
    debug_detections_list()
    all_detections = all_detections[:100]

    return bounding_boxes

def segment_objects(frame, boxes, target_height, target_width):
    """Segments objects into SMALL, MEDIUM, LARGE by coverage."""
    total_image_area = int(target_height) * int(target_width)
    for box in boxes:
        x1, y1 = box['x'], box['y']
        x2, y2 = x1 + box['w'], y1 + box['h']
        actual_area = calculate_actual_object_area(frame, [x1, y1, x2, y2])
        size_category, coverage_percentage = classify_by_coverage(actual_area, total_image_area)
        box['size_category'] = size_category
        box['coverage_pct'] = coverage_percentage
    return boxes

def calculate_actual_object_area(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return (x2 - x1) * (y2 - y1)
    try:
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            actual_area = cv2.contourArea(largest_contour)
            bbox_area = (x2 - x1) * (y2 - y1)
            if actual_area >= bbox_area * 0.1:
                return actual_area
            else:
                return bbox_area * 0.7
        bbox_area = (x2 - x1) * (y2 - y1)
        return bbox_area * 0.6
    except Exception:
        return (x2 - x1) * (y2 - y1)

def classify_by_coverage(area, total_image_area):
    small = 8.0
    large = 25.0
    coverage_percentage = (area / total_image_area) * 100 if total_image_area > 0 else 0.0
    if coverage_percentage <= small:
        return 'small', coverage_percentage
    elif coverage_percentage >= large:
        return 'large', coverage_percentage
    else:
        return 'medium', coverage_percentage

@app.route('/get_log')
def get_log():
    global all_detections, weight_monitoring_enabled
    log_html = "<table class='table-auto w-full border-separate border-spacing-1'>"
    log_html += "<thead><tr class='text-[14px]'><th style='text-align: left;'>Object</th>"
    if weight_monitoring_enabled:
        log_html += "<th style='text-align: right;'>Weight (g)</th>"
    log_html += "<th style='text-align: right;'>Timestamp</th></tr></thead><tbody>"

    if not all_detections:
        colspan = 3 if weight_monitoring_enabled else 2
        log_html += f"<tr><td colspan='{colspan}' class='text-center p-2'>No detections yet.</td></tr>"
        log_html += "</tbody></table>"
        return log_html

    for entry in all_detections:
        if len(entry) == 3:
            counts, timestamp, current_weight = entry
        else:
            counts, timestamp = entry
            current_weight = None

        detection_strings = [f"{count} {obj}s" if count > 1 else f"{count} {obj}"
                             for obj, count in counts.items() if count > 0]
        detections = ", ".join(detection_strings)

        if detections:
            log_html += "<tr>"
            log_html += f"<td class='bg-white p-2 text-[14px]'>{detections}</td>"
            if weight_monitoring_enabled:
                if current_weight is not None and current_weight > 0.00:
                    log_html += f"<td class='bg-white p-2 text-[14px] text-right'>{current_weight:.2f}</td>"
                else:
                    log_html += "<td class='bg-white p-2 text-[14px] text-right'>—</td>"
            log_html += f"<td class='bg-white p-2 text-[14px] text-right'>{timestamp}</td>"
            log_html += "</tr>"

    log_html += "</tbody></table>"
    return log_html

@app.route('/get_object')
def get_object():
    if not all_detections:
        return "<p class='text-[14px] bg-white p-3'>No detections available.</p>"

    if len(all_detections[0]) == 3:
        latest_counts, latest_timestamp, latest_weight = all_detections[0]
    else:
        latest_counts, latest_timestamp = all_detections[0]
        latest_weight = None

    sorted_counts = sorted(latest_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    object_html = "<table class='table-auto w-full border-separate border-spacing-2'>"
    object_html += "<thead class=''><tr class='text-[14px]'><th style='text-align: left;'>Object</th><th style='text-align: right;'>Count</th></tr></thead>"
    object_html += "<tbody>"
    for obj, count in sorted_counts:
        if count > 0:
            object_html += (
                f"<tr>"
                f"<td class='bg-white p-2 text-[14px]'>{obj}</td>"
                f"<td class='bg-white p-2 text-[14px] text-right'>{count}</td>"
                f"</tr>"
            )
    object_html += "</tbody></table>"
    return object_html

@app.route('/download_csv')
def download_csv():
    output = io.BytesIO()
    csv_output = io.StringIO()
    try:
        fieldnames = ['Timestamp'] + list(set(obj for entry in all_detections for obj in entry[0].keys()))
    except Exception as e:
        print(f"[ERROR] Failed to build fieldnames: {e}")
        return jsonify({'error': 'No valid detection data found'}), 500

    writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    writer.writeheader()

    for entry in all_detections:
        try:
            if len(entry) == 3:
                counts, timestamp, current_weight = entry
            else:
                counts, timestamp = entry
                current_weight = None
            row = {'Timestamp': timestamp}
            row.update({obj: counts.get(obj, 0) for obj in fieldnames[1:]})
            writer.writerow(row)
        except Exception as e:
            print(f"[WARN] Skipping invalid detection entry {entry}: {e}")
            continue

    csv_output.seek(0)
    try:
        df = pd.read_csv(io.StringIO(csv_output.getvalue()))
        if df.empty:
            print("[WARN] Empty DataFrame, skipping chart generation.")
            return jsonify({'error': 'No data available for report'}), 200
    except Exception as e:
        print(f"[ERROR] Failed to load CSV into DataFrame: {e}")
        return jsonify({'error': 'Invalid CSV data'}), 500

    plt.figure(figsize=(10, 6))
    for item in df.columns[1:]:
        plt.plot(df['Timestamp'], df[item], marker='o', label=item)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Inventory Usage Trends Over Time')
    plt.legend()
    plt.tight_layout()

    graph_output = io.BytesIO()
    plt.savefig(graph_output, format='png', bbox_inches='tight')
    plt.close('all')
    graph_output.seek(0)

    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Data']
            worksheet.set_column('E:E', 30)
            worksheet.set_row(1, 200)
            worksheet.insert_image('E2', 'graph.png', {'image_data': graph_output, 'x_scale': 0.9, 'y_scale': 0.9})

        output.seek(0)
        return send_file(output,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True,
                         download_name='detections_report.xlsx')
    except Exception as e:
        print(f"[ERROR] Failed to generate Excel report: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500
    finally:
        plt.close('all')

def generate_line_chart(fieldnames, frame_numbers, object_counts_per_frame):
    plt.figure(figsize=(6, 4))
    for field in fieldnames:
        plt.plot(frame_numbers, object_counts_per_frame[field], marker="o", label=field)
    plt.xlabel("Frame Number")
    plt.ylabel("Detections")
    plt.title("Object Detection Trends Over Frames")
    plt.legend()
    report_path = f'{REPORTS_FOLDER}/detection_trend.png'
    plt.savefig(report_path)
    plt.close()
    return report_path

def generate_pie_chart(total_counts):
    plt.figure(figsize=(6, 4))
    if sum(total_counts.values()) == 0:
        total_counts = {"None": 1}
    plt.pie(total_counts.values(), labels=total_counts.keys(), autopct='%1.1f%%',
            colors=["blue", "green", "red", "purple", "orange", "teal"])
    plt.title("Object Detection Distribution")
    report_path = f"{REPORTS_FOLDER}/object_distribution.png"
    plt.savefig(report_path)
    plt.close()
    return report_path

def generate_detection_histogram(fieldnames, object_counts_per_frame):
    plt.figure(figsize=(6, 4))
    for field in fieldnames:
        values = [row.get(field, 0) for row in object_counts_per_frame]
        plt.hist(values, bins=5, alpha=0.6, label=field)
    plt.xlabel("Number of Detections")
    plt.ylabel("Frequency")
    plt.title("Detection Frequency Distribution")
    plt.legend()
    report_path = f"{REPORTS_FOLDER}/detection_histogram.png"
    plt.savefig(report_path)
    plt.close()
    return report_path

def generate_total_detections_chart(total_counts):
    plt.figure(figsize=(6, 4))
    if not total_counts:
        total_counts = {"None": 0}
    plt.bar(total_counts.keys(), total_counts.values(), color=["blue", "green", "red", "purple", "orange", "teal"])
    plt.xlabel("Object")
    plt.ylabel("Total Detections")
    plt.title("Total Object Detections")
    report_path = f"{REPORTS_FOLDER}/total_detections.png"
    plt.savefig(report_path)
    plt.close()
    return report_path

def product_distribution_by_country():
    df = pd.DataFrame(barcode_product_data)
    country_counts = df["countries"].value_counts()
    plt.figure(figsize=(8, 5))
    country_counts.plot(kind="bar", color="lightcoral")
    plt.xlabel("Country")
    plt.ylabel("Number of Products")
    plt.title("Product Distribution by Country")
    plt.xticks(rotation=45)
    output_path = f'{REPORTS_FOLDER}/country_distribution.png'
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_report(mode="OBJECT"):
    stop_detection()
    reports = []
    if mode == "OBJECT":
        all_objects = []
        fieldnames = list(set(obj for entry in all_detections for obj in entry[0].keys()))
        for entry in all_detections:
            if len(entry) == 3:
                counts, timestamp, current_weight = entry
            else:
                counts, timestamp = entry
                current_weight = None
            row = {obj: counts.get(obj, 0) for obj in fieldnames}
            all_objects.append(row)

        total_counts = {field: 0 for field in fieldnames}
        for row in all_objects:
            for field in fieldnames:
                total_counts[field] += row[field]

        frame_numbers = list(range(1, len(all_objects) + 1))
        object_counts_per_frame = {field: [row[field] for row in all_objects] for field in fieldnames}

        line_chart_path = generate_line_chart(fieldnames, frame_numbers, object_counts_per_frame)
        pie_chart = generate_pie_chart(total_counts)
        total_chart = generate_total_detections_chart(total_counts)
        detection_histogram = generate_detection_histogram(fieldnames, all_objects)

        reports = [line_chart_path, pie_chart, total_chart, detection_histogram]
    return reports

@app.route('/stream_reports')
def stream_reports():
    try:
        mode = request.args.get("type", "OBJECT")
        report_images = generate_report(mode.upper())
        if not report_images:
            return jsonify({"message": "No data available for report generation."}), 200
        return jsonify(report_images)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": f"An error occurred: {e}"}), 500

@app.route("/report/<filename>")
def get_report(filename):
    return send_file(os.path.join(REPORTS_FOLDER, filename), mimetype="image/png")

@app.route('/start_detection')
def start_detection():
    global is_detecting, last_detection_time
    if not is_detecting:
        if is_training:
            return jsonify({'status': 'error', 'message': 'Cannot start detection while training is in progress'}), 400
        is_detecting = True
        last_detection_time = time.time()
        return jsonify({'status': 'success', 'message': 'Detection started'})
    return jsonify({'status': 'info', 'message': 'Detection already running'})

@app.route('/stop_detection')
def stop_detection():
    global is_detecting
    if is_detecting:
        is_detecting = False
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    return jsonify({'status': 'info', 'message': 'Detection is not running'})

@app.route('/set_detection_mode', methods=['POST'])
def set_detection_mode():
    global detection_mode
    try:
        detect_mode = request.form.get('detection_mode', 'OBJECT')
        if detect_mode not in ["OBJECT", "OCR", "BARCODE"]:
            return jsonify({'status': 'error', 'message': 'Invalid detection mode'}), 400
        detection_mode = detect_mode
        return jsonify({'status': 'success', 'message': f'Detection mode set to {detect_mode}'})
    except ValueError:
        return jsonify({'status': 'error'})

@app.route('/set_interval', methods=['POST'])
def set_interval():
    global image_save_interval
    try:
        new_interval = float(request.form['interval'])
        if new_interval > 0:
            image_save_interval = new_interval
            return jsonify({'status': 'success', 'message': f'Detection interval set to {new_interval} seconds'})
        else:
            return jsonify({'status': 'error', 'message': 'Interval must be a positive number'})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid interval value'})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(IMAGE_FOLDER, file.filename)
        file.save(filename)
        return redirect(url_for('uploaded_file', filename=file.filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(IMAGE_FOLDER, filename))

@app.route('/retrain')
def retrain():
    """
    GPU training using .pt weights, then auto-export to TensorRT .engine and hot-reload inference model.
    """
    global model, is_training

    if is_training:
        return jsonify({'status': 'error', 'message': 'Training is already in progress'}), 400

    if not os.path.isfile(PT_WEIGHTS):
        return jsonify({'status': 'error', 'message': f'PT weights not found: {PT_WEIGHTS}'}), 400

    is_training = True

    def train_model():
        global model, is_training
        try:
            data_yaml = 'data.yaml'
            if not os.path.isfile(data_yaml):
                print("Training data configuration file not found")
                return

            print("🔧 Starting model training on GPU...")
            training_model = YOLO(PT_WEIGHTS)
            # Ultralytics uses global torch device; we ensure GPU with device parameter in train() if supported
            training_model.train(data=data_yaml, epochs=10, imgsz=IMG_SIZE, device=DEVICE)
            print(" Training completed successfully")

            # Try to export to engine and hot-reload
            try:
                print(" Exporting trained model to TensorRT engine...")
                engine_out = training_model.export(format='engine', device=DEVICE, imgsz=IMG_SIZE)
                # engine_out can be a path string or object; ensure path
                engine_path = str(engine_out) if isinstance(engine_out, (str, Path)) else ENGINE_PATH
                if not os.path.exists(engine_path):
                    engine_path = ENGINE_PATH  # fallback to default name
                print(f" Exported engine: {engine_path}")
                # Hot-reload inference engine
                model = YOLO(engine_path)
                print("  Inference model reloaded with new engine.")
            except Exception as e:
                print(f" Engine export failed, keeping previous engine. Error: {e}")

        except Exception as e:
            print(f" Error during training: {e}")
        finally:
            is_training = False

    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    return jsonify({'status': 'success', 'message': 'Training started in background'}), 200

# for debugging purpose only
def debug_detections_list():
    print("\n[DEBUG] Latest 5 entries in all_detections:")
    for det in all_detections[:5]:
        print(det)
    print("==========================================\n")

if __name__ == '__main__':
    # Disable auto-reloader on Windows to avoid "not a socket" error
    app.run(debug=True, use_reloader=False, port=5001)

