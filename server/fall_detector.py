import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch

from model import FallLSTM


class FallDetector:
    def __init__(self, cfg):
        self.cfg = cfg

        # 데이터 설정
        self.seq_len = cfg["dataset"]["seq_len"]
        self.num_keypoints = cfg["dataset"]["num_keypoints"]
        self.features = cfg["dataset"]["features"]

        # 추론 설정
        inference_cfg = cfg["inference"]

        self.confidence_threshold = inference_cfg[
            "confidence_threshold"
        ]
        self.required_fall_frames = inference_cfg.get(
            "required_fall_frames",
            5,
        )
        self.no_pose_reset_seconds = inference_cfg.get(
            "no_pose_reset_seconds",
            2,
        )

        # 라벨 번호를 이름으로 변환
        self.label_map = {
            label_index: label_name
            for label_name, label_index in cfg["labels"].items()
        }

        # 장치 설정
        self.device = self._get_device()

        # 모델 생성 및 가중치 로딩
        self.model = FallLSTM(cfg).to(self.device)
        self._load_model(cfg["paths"]["model_path"])
        self.model.eval()

        # 키포인트 시퀀스
        self.sequence = deque(maxlen=self.seq_len)

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 현재 분류 상태
        self.current_label = "waiting"
        self.current_confidence = 0.0

        # 넘어짐 경고 상태
        self.fall_detected = False
        self.detected_at = None
        self.fall_count = 0
        self.event_id = 0

        # 마지막 사람 검출 시간
        self.last_pose_time = None

        # 저장 영상 경로
        self.last_video_path = None
        self.last_video_name = None

        # 여러 스레드가 상태에 접근하므로 lock 사용
        self.state_lock = threading.Lock()

    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"[INFO] 추론 장치: {device}")
        return device

    def _load_model(self, model_path):
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {path.resolve()}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        elif (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        print(f"[INFO] 모델 로딩 완료: {path.resolve()}")

    def extract_keypoints(self, results):
        """
        MediaPipe Pose의 33개 관절에서 x, y, z를 추출합니다.
        결과 크기: 33 * 3 = 99
        """
        if results.pose_landmarks is None:
            return None

        keypoints = []

        for landmark in results.pose_landmarks.landmark:
            keypoints.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
            )

        return np.asarray(
            keypoints,
            dtype=np.float32,
        )

    def normalize_keypoints(self, keypoints):
        """
        33개의 x, y, z 좌표를 골반 중심 기준으로 정규화합니다.

        주의:
        학습할 때 다른 normalize_keypoints()를 사용했다면
        반드시 학습 코드와 같은 정규화 함수를 사용해야 합니다.
        """
        points = keypoints.reshape(
            self.num_keypoints,
            self.features,
        ).copy()

        # MediaPipe Pose 기준
        # 왼쪽 골반 23, 오른쪽 골반 24
        left_hip = points[23]
        right_hip = points[24]

        hip_center = (left_hip + right_hip) / 2.0

        points = points - hip_center

        # 어깨 사이 거리를 크기 기준으로 사용
        left_shoulder = points[11]
        right_shoulder = points[12]

        scale = np.linalg.norm(
            left_shoulder[:2] - right_shoulder[:2]
        )

        if scale < 1e-6:
            scale = 1.0

        points = points / scale

        return points.flatten().astype(np.float32)

    def predict(self):
        if len(self.sequence) < self.seq_len:
            return "waiting", 0.0

        sequence_array = np.asarray(
            self.sequence,
            dtype=np.float32,
        )

        expected_shape = (
            self.seq_len,
            self.num_keypoints * self.features,
        )

        if sequence_array.shape != expected_shape:
            raise ValueError(
                "시퀀스 크기가 올바르지 않습니다. "
                f"현재: {sequence_array.shape}, "
                f"예상: {expected_shape}"
            )

        input_tensor = torch.from_numpy(
            sequence_array
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            confidence, label_index = torch.max(
                probabilities,
                dim=1,
            )

        index = int(label_index.item())
        confidence_value = float(confidence.item())

        label = self.label_map.get(
            index,
            f"unknown_{index}",
        )

        return label, confidence_value

    def process_frame(self, frame):
        """
        한 프레임에 대해 관절 추출, LSTM 예측,
        상태 표시까지 수행합니다.
        """
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        now = datetime.now()

        if results.pose_landmarks is not None:
            self.last_pose_time = now

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

            keypoints = self.extract_keypoints(results)

            if keypoints is not None:
                normalized = self.normalize_keypoints(keypoints)
                self.sequence.append(normalized)

            if len(self.sequence) == self.seq_len:
                label, confidence = self.predict()
                self._update_prediction(
                    label,
                    confidence,
                )
        else:
            self._handle_no_pose(now)

        self.draw_status(frame)

        return frame

    def _update_prediction(self, label, confidence):
        with self.state_lock:
            self.current_label = label
            self.current_confidence = confidence

            model_fall = (
                label == "fall"
                and confidence >= self.confidence_threshold
            )

            if model_fall:
                self.fall_count += 1
            else:
                self.fall_count = 0

            if (
                self.fall_count >= self.required_fall_frames
                and not self.fall_detected
            ):
                self.fall_detected = True
                self.detected_at = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.event_id += 1

                print(
                    "[ALERT] 넘어짐 감지:",
                    self.detected_at,
                    f"confidence={confidence:.4f}",
                )

                self._send_discord_async()

    def _handle_no_pose(self, now):
        if self.last_pose_time is None:
            return

        elapsed = (
            now - self.last_pose_time
        ).total_seconds()

        if elapsed >= self.no_pose_reset_seconds:
            self.sequence.clear()

            with self.state_lock:
                self.current_label = "no_person"
                self.current_confidence = 0.0
                self.fall_count = 0

    def draw_status(self, frame):
        with self.state_lock:
            label = self.current_label
            confidence = self.current_confidence
            fall_alert = self.fall_detected

        height, width = frame.shape[:2]

        if fall_alert:
            color = (0, 0, 255)
            text = (
                f"FALL ALERT | "
                f"{label.upper()} {confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (2, 2),
                (width - 3, height - 3),
                color,
                14,
            )

        elif label == "fall":
            color = (0, 100, 255)
            text = f"CHECKING FALL {confidence:.2f}"

        elif label == "lying":
            color = (0, 165, 255)
            text = f"LYING {confidence:.2f}"

        elif label == "sitting":
            color = (255, 255, 0)
            text = f"SITTING {confidence:.2f}"

        elif label == "normal":
            color = (0, 255, 0)
            text = f"NORMAL {confidence:.2f}"

        elif label == "no_person":
            color = (180, 180, 180)
            text = "NO PERSON"

        else:
            color = (255, 255, 255)
            text = "WAITING FOR SEQUENCE"

        cv2.rectangle(
            frame,
            (10, 10),
            (min(width - 10, 520), 58),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    def get_status(self):
        with self.state_lock:
            return {
                "fall_detected": self.fall_detected,
                "label": self.current_label,
                "confidence": round(
                    self.current_confidence,
                    4,
                ),
                "detected_at": self.detected_at,
                "event_id": self.event_id,
                "fall_count": self.fall_count,
                "required_fall_frames": (
                    self.required_fall_frames
                ),
                "last_video_name": self.last_video_name,
            }

    def reset_fall(self):
        """
        앱의 확인 버튼으로 넘어짐 경고만 해제합니다.
        자세 분류 시퀀스는 유지합니다.
        """
        with self.state_lock:
            self.fall_detected = False
            self.detected_at = None
            self.fall_count = 0

    def set_saved_video(self, video_path):
        if video_path is None:
            return

        with self.state_lock:
            self.last_video_path = str(video_path)
            self.last_video_name = Path(video_path).name

    def _send_discord_async(self):
        if not self.cfg["discord"].get("enable", False):
            return

        thread = threading.Thread(
            target=self._send_discord,
            daemon=True,
        )
        thread.start()

    def _send_discord(self):
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        if not webhook_url:
            print(
                "[WARNING] DISCORD_WEBHOOK_URL "
                "환경변수가 설정되지 않았습니다."
            )
            return

        with self.state_lock:
            detected_at = self.detected_at
            confidence = self.current_confidence
            label = self.current_label

        message = {
            "content": (
                "🚨 넘어짐이 감지되었습니다.\n"
                f"- 감지 시간: {detected_at}\n"
                f"- 현재 자세: {label}\n"
                f"- 신뢰도: {confidence * 100:.1f}%"
            )
        }

        try:
            response = requests.post(
                webhook_url,
                json=message,
                timeout=5,
            )
            response.raise_for_status()

            print("[INFO] Discord 알림 전송 완료")

        except requests.RequestException as error:
            print(
                f"[WARNING] Discord 알림 실패: {error}"
            )

    def close(self):
        self.pose.close()