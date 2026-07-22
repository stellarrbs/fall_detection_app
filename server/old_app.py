import atexit
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    send_from_directory,
)
from flask_cors import CORS

from configs.config_loader import load_config
from fall_detector import FallDetector


cfg = load_config("configs/config.yaml")

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

camera_index = cfg["inference"]["camera_index"]
frame_width = cfg["inference"].get(
    "frame_width",
    640,
)
frame_height = cfg["inference"].get(
    "frame_height",
    480,
)
jpeg_quality = cfg["inference"].get(
    "jpeg_quality",
    75,
)

recording_cfg = cfg["recording"]
recording_enabled = recording_cfg.get(
    "enable",
    False,
)
recording_fps = recording_cfg.get(
    "fps",
    20,
)
pre_record_seconds = recording_cfg.get(
    "pre_record_seconds",
    5,
)
post_record_seconds = recording_cfg.get(
    "post_record_seconds",
    10,
)

alert_video_dir = Path(
    cfg["paths"]["alert_video_dir"]
)
alert_video_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 카메라 및 감지기
# --------------------------------------------------

camera = cv2.VideoCapture(camera_index)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    frame_width,
)
camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    frame_height,
)
camera.set(
    cv2.CAP_PROP_FPS,
    recording_fps,
)

detector = FallDetector(cfg)


# --------------------------------------------------
# 공유 상태
# --------------------------------------------------

latest_jpeg = None
latest_frame_time = 0.0
camera_connected = False
server_started_at = time.time()

frame_lock = threading.Lock()

# 넘어짐 이전 프레임 저장
pre_buffer_size = max(
    1,
    int(recording_fps * pre_record_seconds),
)

pre_frame_buffer = deque(
    maxlen=pre_buffer_size,
)

# 저장 상태
recording_active = False
recording_writer = None
recording_end_time = 0.0
recording_path = None
recorded_event_id = 0


# --------------------------------------------------
# 영상 저장 함수
# --------------------------------------------------

def start_alert_recording(frame, event_id):
    global recording_active
    global recording_writer
    global recording_end_time
    global recording_path
    global recorded_event_id

    if not recording_enabled:
        return

    if recording_active:
        return

    height, width = frame.shape[:2]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    file_name = f"fall_{timestamp}.mp4"
    recording_path = alert_video_dir / file_name

    # macOS와 모바일에서 호환성이 좋은 mp4v 사용
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    recording_writer = cv2.VideoWriter(
        str(recording_path),
        fourcc,
        recording_fps,
        (width, height),
    )

    if not recording_writer.isOpened():
        print(
            "[WARNING] 저장 영상 파일을 열 수 없습니다:",
            recording_path,
        )
        recording_writer = None
        recording_path = None
        return

    # 넘어짐 감지 전 버퍼 먼저 기록
    for buffered_frame in list(pre_frame_buffer):
        recording_writer.write(buffered_frame)

    recording_active = True
    recording_end_time = (
        time.time() + post_record_seconds
    )
    recorded_event_id = event_id

    print(
        "[INFO] 넘어짐 영상 저장 시작:",
        recording_path,
    )


def write_recording_frame(frame):
    global recording_active
    global recording_writer
    global recording_path

    if not recording_active:
        return

    if recording_writer is None:
        recording_active = False
        return

    recording_writer.write(frame)

    if time.time() >= recording_end_time:
        recording_writer.release()

        saved_path = recording_path

        recording_writer = None
        recording_active = False
        recording_path = None

        detector.set_saved_video(saved_path)

        print(
            "[INFO] 넘어짐 영상 저장 완료:",
            saved_path,
        )


# --------------------------------------------------
# 카메라 반복 스레드
# --------------------------------------------------

def camera_loop():
    global latest_jpeg
    global latest_frame_time
    global camera_connected

    while True:
        success, frame = camera.read()

        if not success or frame is None:
            camera_connected = False
            time.sleep(0.1)
            continue

        camera_connected = True
        latest_frame_time = time.time()

        processed_frame = detector.process_frame(
            frame
        )

        # 감지 전 영상 버퍼에 프레임 저장
        if recording_enabled:
            pre_frame_buffer.append(
                processed_frame.copy()
            )

        detector_status = detector.get_status()
        event_id = detector_status["event_id"]

        # 새 넘어짐 사건이 감지되면 저장 시작
        if (
            detector_status["fall_detected"]
            and event_id > recorded_event_id
        ):
            start_alert_recording(
                processed_frame,
                event_id,
            )

        write_recording_frame(processed_frame)

        encode_success, buffer = cv2.imencode(
            ".jpg",
            processed_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality,
            ],
        )

        if not encode_success:
            continue

        with frame_lock:
            latest_jpeg = buffer.tobytes()


def generate_mjpeg():
    while True:
        with frame_lock:
            jpeg = latest_jpeg

        if jpeg is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n"
            + jpeg
            + b"\r\n"
        )

        # 약 20FPS
        time.sleep(0.05)


# --------------------------------------------------
# Flask API
# --------------------------------------------------

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >
        <title>넘어짐 감지 카메라</title>

        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                background: #111827;
                color: white;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            h2 {
                margin: 18px 0;
            }

            .camera-box {
                width: min(100%, 900px);
                background: black;
            }

            .camera-box img {
                display: block;
                width: 100%;
                height: auto;
            }

            #status {
                width: min(100%, 900px);
                padding: 18px;
                background: #1f2937;
                font-size: 18px;
            }

            .normal {
                color: #4ade80;
            }

            .fall {
                color: #f87171;
                font-weight: bold;
            }

            .error {
                color: #f87171;
            }
        </style>
    </head>

    <body>
        <h2>넘어짐 감지 카메라</h2>

        <div class="camera-box">
            <img src="/video" alt="실시간 카메라 영상">
        </div>

        <div id="status">
            상태를 확인하는 중입니다.
        </div>

        <script>
            async function updateStatus() {
                const statusElement =
                    document.getElementById("status");

                try {
                    const response = await fetch("/status");
                    const data = await response.json();

                    if (!data.camera_connected) {
                        statusElement.className = "error";
                        statusElement.innerText =
                            "카메라 연결 끊김";
                        return;
                    }

                    if (data.fall_detected) {
                        statusElement.className = "fall";
                        statusElement.innerText =
                            "⚠ 넘어짐 감지 | 시간: " +
                            (data.detected_at ?? "-") +
                            " | 현재 자세: " +
                            data.label +
                            " | 신뢰도: " +
                            (data.confidence * 100)
                                .toFixed(1) +
                            "%";
                    } else {
                        statusElement.className = "normal";
                        statusElement.innerText =
                            "● 카메라 연결됨 | 현재 자세: " +
                            data.label +
                            " | 신뢰도: " +
                            (data.confidence * 100)
                                .toFixed(1) +
                            "%";
                    }
                } catch (error) {
                    statusElement.className = "error";
                    statusElement.innerText =
                        "서버 상태를 불러올 수 없습니다.";
                }
            }

            setInterval(updateStatus, 1000);
            updateStatus();
        </script>
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_mjpeg(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


@app.route("/status")
def status():
    if latest_frame_time > 0:
        frame_age = time.time() - latest_frame_time
    else:
        frame_age = None

    is_camera_connected = (
        camera_connected
        and frame_age is not None
        and frame_age < 3
    )

    detector_status = detector.get_status()

    video_url = None

    if detector_status["last_video_name"]:
        video_url = (
            "/alert-videos/"
            + detector_status["last_video_name"]
        )

    return jsonify(
        {
            "server_connected": True,
            "camera_connected": (
                is_camera_connected
            ),
            "frame_age": (
                round(frame_age, 3)
                if frame_age is not None
                else None
            ),
            "recording_active": recording_active,
            "video_url": video_url,
            "uptime_seconds": round(
                time.time() - server_started_at,
                1,
            ),
            **detector_status,
        }
    )


@app.route("/reset-fall", methods=["POST"])
def reset_fall():
    detector.reset_fall()

    return jsonify(
        {
            "success": True,
            "message": "넘어짐 경고가 초기화되었습니다.",
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "running",
            "camera_opened": camera.isOpened(),
        }
    )


@app.route(
    "/alert-videos/<path:file_name>"
)
def alert_video(file_name):
    return send_from_directory(
        alert_video_dir.resolve(),
        file_name,
        as_attachment=False,
    )


# --------------------------------------------------
# 종료 처리
# --------------------------------------------------

def cleanup():
    global recording_writer

    print("[INFO] 서버 자원을 정리합니다.")

    if recording_writer is not None:
        recording_writer.release()
        recording_writer = None

    if camera.isOpened():
        camera.release()

    detector.close()


atexit.register(cleanup)


# --------------------------------------------------
# 서버 시작
# --------------------------------------------------

if __name__ == "__main__":
    if not camera.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다.\n"
            "1. macOS 카메라 권한을 확인하세요.\n"
            "2. 다른 프로그램이 카메라를 사용 중인지 확인하세요.\n"
            "3. config.yaml의 camera_index를 확인하세요."
        )

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True,
    )
    camera_thread.start()

    host = cfg.get(
        "server",
        {},
    ).get(
        "host",
        "0.0.0.0",
    )

    port = cfg.get(
        "server",
        {},
    ).get(
        "port",
        5000,
    )

    try:
        app.run(
            host=host,
            port=port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
    finally:
        cleanup()