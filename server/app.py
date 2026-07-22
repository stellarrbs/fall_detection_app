import atexit
import copy
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

from configs.config_loader import load_config
from fall_detector import FallDetector


# ==================================================
# Flask 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"

app = Flask(__name__)
CORS(app)


# ==================================================
# 전역 상태
# ==================================================

cfg = None
camera: Optional[cv2.VideoCapture] = None
detector: Optional[FallDetector] = None

camera_thread: Optional[threading.Thread] = None
stop_event = threading.Event()

latest_jpeg: Optional[bytes] = None
latest_frame_time = 0.0
camera_connected = False
camera_error: Optional[str] = None
server_started_at = time.time()

frame_lock = threading.Lock()
state_lock = threading.Lock()
recording_lock = threading.Lock()

frame_width = 640
frame_height = 480
jpeg_quality = 75

recording_enabled = False
recording_fps = 20
pre_record_seconds = 5
post_record_seconds = 10

alert_video_dir = BASE_DIR / "alert_videos"
pre_frame_buffer = deque(maxlen=100)

recording_active = False
recording_writer: Optional[cv2.VideoWriter] = None
recording_end_time = 0.0
recording_path: Optional[Path] = None
recorded_event_id = 0

_cleanup_done = False


# ==================================================
# 초기화
# ==================================================

def resolve_config_paths(config: dict) -> dict:
    """
    config.yaml의 상대 경로를 app.py가 있는 server 폴더 기준의
    절대 경로로 변경합니다.
    """
    resolved = copy.deepcopy(config)

    paths = resolved.setdefault("paths", {})

    for key in ("model_dir", "model_path", "alert_video_dir"):
        value = paths.get(key)

        if not value:
            continue

        path = Path(value)

        if not path.is_absolute():
            path = BASE_DIR / path

        paths[key] = str(path.resolve())

    return resolved


def initialize_components() -> None:
    """
    설정, 저장 폴더, 감지 모델, 카메라를 순서대로 초기화합니다.
    초기화 진행 상황을 터미널에 출력합니다.
    """
    global cfg
    global camera
    global detector
    global frame_width
    global frame_height
    global jpeg_quality
    global recording_enabled
    global recording_fps
    global pre_record_seconds
    global post_record_seconds
    global alert_video_dir
    global pre_frame_buffer

    print("[1/5] 설정 파일을 읽습니다.", flush=True)

    loaded_cfg = load_config(str(CONFIG_PATH))
    cfg = resolve_config_paths(loaded_cfg)

    print(f"[INFO] 설정 파일: {CONFIG_PATH}", flush=True)

    inference_cfg = cfg.get("inference", {})
    recording_cfg = cfg.get("recording", {})

    frame_width = int(inference_cfg.get("frame_width", 640))
    frame_height = int(inference_cfg.get("frame_height", 480))
    jpeg_quality = int(inference_cfg.get("jpeg_quality", 75))

    recording_enabled = bool(recording_cfg.get("enable", False))
    recording_fps = int(recording_cfg.get("fps", 20))
    pre_record_seconds = int(
        recording_cfg.get("pre_record_seconds", 5)
    )
    post_record_seconds = int(
        recording_cfg.get("post_record_seconds", 10)
    )

    alert_video_dir = Path(
        cfg["paths"].get(
            "alert_video_dir",
            str(BASE_DIR / "alert_videos"),
        )
    )
    alert_video_dir.mkdir(parents=True, exist_ok=True)

    pre_buffer_size = max(
        1,
        recording_fps * pre_record_seconds,
    )
    pre_frame_buffer = deque(maxlen=pre_buffer_size)

    print("[2/5] 넘어짐 감지 모델을 불러옵니다.", flush=True)

    detector = FallDetector(cfg)

    print("[INFO] 감지 모델 준비 완료", flush=True)
    print("[3/5] 카메라를 엽니다.", flush=True)

    camera_index = int(inference_cfg.get("camera_index", 0))

    # macOS에서는 AVFoundation을 명시하면 카메라 연결이 안정적인 경우가 많습니다.
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_AVFOUNDATION,
        )
    else:
        camera = cv2.VideoCapture(camera_index)

    if camera is None or not camera.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다.\n"
            "1. macOS 시스템 설정 → 개인정보 보호 및 보안 → "
            "카메라에서 Terminal 또는 VS Code 권한을 허용하세요.\n"
            "2. FaceTime, Zoom 등 카메라를 사용하는 앱을 종료하세요.\n"
            "3. config.yaml의 camera_index를 확인하세요."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    camera.set(cv2.CAP_PROP_FPS, recording_fps)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[INFO] 카메라 열기 성공: index={camera_index}, "
        f"{actual_width}x{actual_height}",
        flush=True,
    )

    print("[4/5] 카메라 스레드를 시작합니다.", flush=True)

    start_camera_thread()

    print("[INFO] 카메라 스레드 시작 완료", flush=True)


# ==================================================
# 영상 저장
# ==================================================

def start_alert_recording(frame, event_id: int) -> None:
    global recording_active
    global recording_writer
    global recording_end_time
    global recording_path
    global recorded_event_id

    if not recording_enabled:
        recorded_event_id = max(recorded_event_id, event_id)
        return

    with recording_lock:
        if recording_active:
            return

        height, width = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_name = f"fall_{timestamp}.mp4"
        path = alert_video_dir / file_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            recording_fps,
            (width, height),
        )

        if not writer.isOpened():
            print(
                f"[WARNING] 저장 영상 파일을 열 수 없습니다: {path}",
                flush=True,
            )
            recorded_event_id = max(recorded_event_id, event_id)
            return

        for buffered_frame in list(pre_frame_buffer):
            writer.write(buffered_frame)

        recording_writer = writer
        recording_path = path
        recording_active = True
        recording_end_time = time.time() + post_record_seconds
        recorded_event_id = event_id

        print(
            f"[INFO] 넘어짐 영상 저장 시작: {path}",
            flush=True,
        )


def write_recording_frame(frame) -> None:
    global recording_active
    global recording_writer
    global recording_path

    saved_path = None

    with recording_lock:
        if not recording_active or recording_writer is None:
            return

        recording_writer.write(frame)

        if time.time() < recording_end_time:
            return

        recording_writer.release()
        saved_path = recording_path

        recording_writer = None
        recording_path = None
        recording_active = False

    if saved_path is not None and detector is not None:
        detector.set_saved_video(saved_path)

        print(
            f"[INFO] 넘어짐 영상 저장 완료: {saved_path}",
            flush=True,
        )


# ==================================================
# 카메라 스레드
# ==================================================

def camera_loop() -> None:
    global latest_jpeg
    global latest_frame_time
    global camera_connected
    global camera_error

    print("[CAMERA] 프레임 처리를 시작합니다.", flush=True)

    while not stop_event.is_set():
        try:
            if camera is None or detector is None:
                raise RuntimeError(
                    "카메라 또는 감지기가 초기화되지 않았습니다."
                )

            success, frame = camera.read()

            if not success or frame is None:
                with state_lock:
                    camera_connected = False
                    camera_error = "카메라 프레임 읽기 실패"

                time.sleep(0.2)
                continue

            processed_frame = detector.process_frame(frame)

            with state_lock:
                camera_connected = True
                camera_error = None
                latest_frame_time = time.time()

            if recording_enabled:
                pre_frame_buffer.append(
                    processed_frame.copy()
                )

            detector_status = detector.get_status()
            event_id = int(detector_status.get("event_id", 0))

            if (
                detector_status.get("fall_detected", False)
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
                raise RuntimeError("JPEG 인코딩에 실패했습니다.")

            with frame_lock:
                latest_jpeg = buffer.tobytes()

        except Exception as error:
            with state_lock:
                camera_connected = False
                camera_error = str(error)

            print(
                f"[CAMERA LOOP ERROR] "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

            time.sleep(0.5)

    print("[CAMERA] 프레임 처리를 종료합니다.", flush=True)


def start_camera_thread() -> None:
    global camera_thread

    camera_thread = threading.Thread(
        target=camera_loop,
        name="camera-loop",
        daemon=True,
    )
    camera_thread.start()


def generate_mjpeg():
    while not stop_event.is_set():
        with frame_lock:
            jpeg = latest_jpeg

        if jpeg is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
            b"Pragma: no-cache\r\n"
            b"Expires: 0\r\n\r\n"
            + jpeg
            + b"\r\n"
        )

        time.sleep(0.05)


# ==================================================
# 웹 페이지 및 API
# ==================================================

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
        }

        .container {
            width: min(100%, 900px);
            margin: 0 auto;
            padding: 16px;
        }

        h2 {
            text-align: center;
        }

        .camera-box {
            width: 100%;
            min-height: 260px;
            background: black;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .camera-box img {
            display: block;
            width: 100%;
            height: auto;
        }

        #status {
            margin-top: 12px;
            padding: 18px;
            border-radius: 12px;
            background: #1f2937;
            font-size: 18px;
        }

        .normal {
            color: #4ade80;
        }

        .fall,
        .error {
            color: #f87171;
            font-weight: bold;
        }

        button {
            width: 100%;
            margin-top: 12px;
            padding: 14px;
            border: 0;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
        }
    </style>
</head>

<body>
    <div class="container">
        <h2>넘어짐 감지 카메라</h2>

        <div class="camera-box">
            <img
                src="/video"
                alt="실시간 카메라 영상"
                onerror="this.style.display='none';"
            >
        </div>

        <div id="status">
            서버 상태를 확인하는 중입니다.
        </div>

        <button onclick="resetFall()">
            넘어짐 경고 확인
        </button>
    </div>

    <script>
        const statusElement =
            document.getElementById("status");

        async function updateStatus() {
            try {
                const response = await fetch(
                    "/status",
                    { cache: "no-store" }
                );

                if (!response.ok) {
                    throw new Error(
                        "HTTP " + response.status
                    );
                }

                const data = await response.json();

                if (!data.camera_connected) {
                    statusElement.className = "error";
                    statusElement.innerText =
                        "카메라 연결 끊김: " +
                        (data.camera_error ?? "원인 확인 중");
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
                    return;
                }

                statusElement.className = "normal";
                statusElement.innerText =
                    "● 영상 수신 중 | 현재 자세: " +
                    data.label +
                    " | 신뢰도: " +
                    (data.confidence * 100)
                        .toFixed(1) +
                    "%";
            } catch (error) {
                statusElement.className = "error";
                statusElement.innerText =
                    "서버 상태를 불러올 수 없습니다: " +
                    error.message;
            }
        }

        async function resetFall() {
            try {
                await fetch(
                    "/reset-fall",
                    {
                        method: "POST",
                        cache: "no-store"
                    }
                );

                await updateStatus();
            } catch (error) {
                alert(
                    "경고 초기화에 실패했습니다: " +
                    error.message
                );
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
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/status")
def status():
    with state_lock:
        current_frame_time = latest_frame_time
        current_camera_connected = camera_connected
        current_camera_error = camera_error

    frame_age = (
        time.time() - current_frame_time
        if current_frame_time > 0
        else None
    )

    is_camera_connected = (
        current_camera_connected
        and frame_age is not None
        and frame_age < 3
    )

    if detector is None:
        detector_status = {
            "fall_detected": False,
            "label": "initializing",
            "confidence": 0.0,
            "detected_at": None,
            "event_id": 0,
            "fall_count": 0,
            "required_fall_frames": 0,
            "last_video_name": None,
        }
    else:
        detector_status = detector.get_status()

    last_video_name = detector_status.get(
        "last_video_name"
    )

    video_url = (
        f"/alert-videos/{last_video_name}"
        if last_video_name
        else None
    )

    return jsonify(
        {
            "server_connected": True,
            "camera_connected": is_camera_connected,
            "camera_error": current_camera_error,
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
    if detector is None:
        return jsonify(
            {
                "success": False,
                "message": "감지기가 아직 준비되지 않았습니다.",
            }
        ), 503

    detector.reset_fall()

    return jsonify(
        {
            "success": True,
            "message": "넘어짐 경고가 초기화되었습니다.",
        }
    )


@app.route("/health")
def health():
    opened = (
        camera is not None
        and camera.isOpened()
    )

    alive = (
        camera_thread is not None
        and camera_thread.is_alive()
    )

    return jsonify(
        {
            "status": "running",
            "camera_opened": opened,
            "camera_thread_alive": alive,
            "detector_ready": detector is not None,
        }
    )


@app.route("/alert-videos/<path:file_name>")
def alert_video(file_name):
    return send_from_directory(
        str(alert_video_dir.resolve()),
        file_name,
        as_attachment=False,
    )


# ==================================================
# 종료 처리
# ==================================================

def cleanup() -> None:
    global _cleanup_done
    global recording_writer

    if _cleanup_done:
        return

    _cleanup_done = True
    stop_event.set()

    print("[INFO] 서버 자원을 정리합니다.", flush=True)

    with recording_lock:
        if recording_writer is not None:
            recording_writer.release()
            recording_writer = None

    if camera is not None and camera.isOpened():
        camera.release()

    if detector is not None:
        detector.close()


atexit.register(cleanup)


# ==================================================
# 실행
# ==================================================

def main() -> None:
    try:
        initialize_components()

        server_cfg = cfg.get("server", {}) if cfg else {}
        host = server_cfg.get("host", "0.0.0.0")
        port = int(server_cfg.get("port", 5001))

        print("[5/5] Flask 서버를 시작합니다.", flush=True)
        print(
            f"[INFO] 컴퓨터 브라우저: "
            f"http://127.0.0.1:{port}",
            flush=True,
        )

        app.run(
            host=host,
            port=port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    except KeyboardInterrupt:
        print("\n[INFO] 사용자가 서버를 종료했습니다.", flush=True)

    except Exception as error:
        print(
            f"\n[FATAL ERROR] "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        raise

    finally:
        cleanup()


if __name__ == "__main__":
    main()
