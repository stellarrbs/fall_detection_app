import os
import threading
import time

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

        # ==================================================
        # 데이터 설정
        # ==================================================

        self.seq_len = int(
            cfg["dataset"]["seq_len"]
        )

        self.num_keypoints = int(
            cfg["dataset"]["num_keypoints"]
        )

        self.features = int(
            cfg["dataset"]["features"]
        )

        # ==================================================
        # 추론 설정
        # ==================================================

        inference_cfg = cfg["inference"]

        self.confidence_threshold = float(
            inference_cfg["confidence_threshold"]
        )

        self.motion_threshold = float(
            inference_cfg.get(
                "motion_threshold",
                0.015,
            )
        )

        self.no_motion_seconds = float(
            inference_cfg.get(
                "no_motion_seconds",
                5,
            )
        )

        self.required_fall_frames = int(
            inference_cfg.get(
                "required_fall_frames",
                3,
            )
        )

        self.no_pose_reset_seconds = float(
            inference_cfg.get(
                "no_pose_reset_seconds",
                2,
            )
        )

        self.velocity_threshold = float(
            inference_cfg.get(
                "velocity_threshold",
                0.8,
            )
        )

        self.acceleration_threshold = float(
            inference_cfg.get(
                "acceleration_threshold",
                8.0,
            )
        )

        self.kinematic_memory_seconds = float(
            inference_cfg.get(
                "kinematic_memory_seconds",
                1.0,
            )
        )

        # 골반과 몸통이 충분히 검출되었는지 판단하는 기준
        self.visibility_threshold = float(
            inference_cfg.get(
                "visibility_threshold",
                0.5,
            )
        )

        # 넘어짐 후보 상태가 너무 오래 유지되는 상황 방지
        self.candidate_timeout = max(
            self.no_motion_seconds + 5.0,
            8.0,
        )

        # ==================================================
        # 라벨 설정
        # ==================================================

        self.label_map = {
            label_index: label_name
            for label_name, label_index
            in cfg["labels"].items()
        }

        # ==================================================
        # 장치 및 모델 설정
        # ==================================================

        self.device = self._get_device()

        self.model = FallLSTM(
            cfg
        ).to(self.device)

        self._load_model(
            cfg["paths"]["model_path"]
        )

        self.model.eval()

        # ==================================================
        # LSTM 입력 시퀀스
        # ==================================================

        self.sequence = deque(
            maxlen=self.seq_len
        )

        # ==================================================
        # 움직임 및 골반 위치 기록
        # ==================================================

        self.motion_history = deque(
            maxlen=60
        )

        self.hip_y_history = deque(
            maxlen=5
        )

        self.previous_raw_keypoints = None

        # ==================================================
        # 수직 속도 및 수직 가속도 상태
        # ==================================================

        self.vertical_velocity = 0.0

        self.vertical_acceleration = 0.0

        self.velocity_event_time = None

        self.acceleration_event_time = None

        # ==================================================
        # 자세 상태
        # ==================================================

        self.torso_horizontal = False

        self.torso_upright = False

        self.torso_visibility = 0.0

        self.lower_body_visibility = 0.0

        self.recent_motion = 0.0

        # ==================================================
        # MediaPipe Pose
        # ==================================================

        self.mp_pose = mp.solutions.pose

        self.mp_drawing = (
            mp.solutions.drawing_utils
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ==================================================
        # 현재 분류 상태
        # ==================================================

        self.current_label = "waiting"

        self.current_confidence = 0.0

        # ==================================================
        # 넘어짐 후보 및 최종 경고 상태
        # ==================================================

        self.fall_candidate_active = False

        self.fall_candidate_start_time = None

        self.fall_detected = False

        self.detected_at = None

        self.fall_count = 0

        self.event_id = 0

        # 마지막 사람 검출 시각
        self.last_pose_time = None

        # 저장된 영상 정보
        self.last_video_path = None

        self.last_video_name = None

        # 여러 스레드가 상태에 접근하므로 lock 사용
        self.state_lock = threading.Lock()

    # ==================================================
    # 장치 및 모델
    # ==================================================

    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device(
                "cuda"
            )

        elif (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends.mps.is_available()
        ):
            device = torch.device(
                "mps"
            )

        else:
            device = torch.device(
                "cpu"
            )

        print(
            f"[INFO] 추론 장치: {device}"
        )

        return device

    def _load_model(self, model_path):
        path = Path(
            model_path
        )

        if not path.exists():
            raise FileNotFoundError(
                "모델 파일을 찾을 수 없습니다: "
                f"{path.resolve()}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "model_state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif (
            isinstance(
                checkpoint,
                dict,
            )
            and "state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        print(
            "[INFO] 모델 로딩 완료: "
            f"{path.resolve()}"
        )

    # ==================================================
    # 관절 좌표 추출 및 정규화
    # ==================================================

    def extract_keypoints(self, results):
        """
        기존 학습 모델과의 호환을 위해
        MediaPipe 관절의 x, y, z 좌표를 유지합니다.

        반환 형태:
            (33, 3)
        """

        if results.pose_landmarks is None:
            return None

        keypoints = []

        for landmark in (
            results.pose_landmarks.landmark
        ):
            keypoints.append(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
            )

        points = np.asarray(
            keypoints,
            dtype=np.float32,
        )

        expected_shape = (
            self.num_keypoints,
            self.features,
        )

        if points.shape != expected_shape:
            raise ValueError(
                "관절 좌표 크기가 올바르지 않습니다. "
                f"현재: {points.shape}, "
                f"예상: {expected_shape}"
            )

        return points

    def extract_visibility(self, results):
        """
        visibility는 LSTM 입력 특징에 포함하지 않고,
        골반 및 몸통 검출 신뢰도 확인에만 사용합니다.
        """

        if results.pose_landmarks is None:
            return None

        visibility = [
            landmark.visibility
            for landmark
            in results.pose_landmarks.landmark
        ]

        return np.asarray(
            visibility,
            dtype=np.float32,
        )

    def normalize_keypoints(self, keypoints):
        """
        기존 fall_detector.py의 정규화 방식을 유지합니다.

        1. 골반 중심을 원점으로 이동
        2. 양쪽 어깨 사이 거리로 크기 정규화
        3. (33, 3)을 길이 99의 배열로 변환
        """

        points = keypoints.reshape(
            self.num_keypoints,
            self.features,
        ).copy()

        left_hip = points[23]

        right_hip = points[24]

        hip_center = (
            left_hip + right_hip
        ) / 2.0

        points = points - hip_center

        left_shoulder = points[11]

        right_shoulder = points[12]

        scale = np.linalg.norm(
            left_shoulder[:2]
            - right_shoulder[:2]
        )

        if scale < 1e-6:
            scale = 1.0

        points = points / scale

        return points.flatten().astype(
            np.float32
        )

    # ==================================================
    # 골반 위치, 속도 및 가속도
    # ==================================================

    def _is_hip_visible(self, visibility):
        if visibility is None:
            return False

        left_hip_visibility = (
            visibility[23]
        )

        right_hip_visibility = (
            visibility[24]
        )

        return bool(
            left_hip_visibility
            >= self.visibility_threshold

            and right_hip_visibility
            >= self.visibility_threshold
        )

    def _get_hip_center_y(
        self,
        raw_keypoints,
    ):
        left_hip_y = (
            raw_keypoints[23, 1]
        )

        right_hip_y = (
            raw_keypoints[24, 1]
        )

        return float(
            (
                left_hip_y
                + right_hip_y
            ) / 2.0
        )

    def _calculate_hip_kinematics(self):
        """
        최근 5개 프레임의 골반 중심 y좌표를 이용해
        수직 속도와 수직 가속도를 계산합니다.

        MediaPipe 좌표계에서는
        y가 증가할수록 화면 아래 방향입니다.
        """

        if len(
            self.hip_y_history
        ) < 5:
            return 0.0, 0.0

        history = list(
            self.hip_y_history
        )[-5:]

        t0, y0 = history[0]

        t2, y2 = history[2]

        t4, y4 = history[4]

        previous_dt = (
            t2 - t0
        )

        current_dt = (
            t4 - t2
        )

        if (
            previous_dt <= 0
            or current_dt <= 0
        ):
            return 0.0, 0.0

        previous_velocity = (
            y2 - y0
        ) / previous_dt

        current_velocity = (
            y4 - y2
        ) / current_dt

        previous_velocity_time = (
            t0 + t2
        ) / 2.0

        current_velocity_time = (
            t2 + t4
        ) / 2.0

        velocity_dt = (
            current_velocity_time
            - previous_velocity_time
        )

        if velocity_dt <= 0:
            return (
                float(
                    current_velocity
                ),
                0.0,
            )

        vertical_acceleration = (
            current_velocity
            - previous_velocity
        ) / velocity_dt

        return (
            float(
                current_velocity
            ),
            float(
                vertical_acceleration
            ),
        )

    def _update_hip_kinematics(
        self,
        raw_keypoints,
        visibility,
        current_time,
    ):
        if not self._is_hip_visible(
            visibility
        ):
            self.hip_y_history.clear()

            self.vertical_velocity = 0.0

            self.vertical_acceleration = 0.0

            return

        hip_center_y = (
            self._get_hip_center_y(
                raw_keypoints
            )
        )

        self.hip_y_history.append(
            (
                current_time,
                hip_center_y,
            )
        )

        (
            self.vertical_velocity,
            self.vertical_acceleration,
        ) = (
            self._calculate_hip_kinematics()
        )

        if (
            self.vertical_velocity
            >= self.velocity_threshold
        ):
            self.velocity_event_time = (
                current_time
            )

        if (
            self.vertical_acceleration
            >= self.acceleration_threshold
        ):
            self.acceleration_event_time = (
                current_time
            )

    def _has_recent_rapid_fall_motion(
        self,
        current_time,
    ):
        velocity_recent = (
            self.velocity_event_time
            is not None

            and (
                current_time
                - self.velocity_event_time
            )
            <= self.kinematic_memory_seconds
        )

        acceleration_recent = (
            self.acceleration_event_time
            is not None

            and (
                current_time
                - self.acceleration_event_time
            )
            <= self.kinematic_memory_seconds
        )

        return bool(
            velocity_recent
            or acceleration_recent
        )

    # ==================================================
    # 자세와 움직임 확인
    # ==================================================

    def _get_body_information(
        self,
        raw_keypoints,
        visibility,
    ):
        torso_indices = [
            11,
            12,
            23,
            24,
        ]

        lower_body_indices = [
            23,
            24,
            25,
            26,
            27,
            28,
        ]

        torso_visibility = float(
            np.mean(
                visibility[
                    torso_indices
                ]
            )
        )

        lower_body_visibility = float(
            np.mean(
                visibility[
                    lower_body_indices
                ]
            )
        )

        shoulder_center = np.mean(
            raw_keypoints[
                [11, 12],
                :2,
            ],
            axis=0,
        )

        hip_center = np.mean(
            raw_keypoints[
                [23, 24],
                :2,
            ],
            axis=0,
        )

        horizontal_distance = abs(
            shoulder_center[0]
            - hip_center[0]
        )

        vertical_distance = abs(
            shoulder_center[1]
            - hip_center[1]
        )

        torso_horizontal = (
            horizontal_distance
            > vertical_distance * 0.8

            and torso_visibility
            >= self.visibility_threshold
        )

        torso_upright = (
            vertical_distance
            > horizontal_distance * 1.2

            and torso_visibility
            >= self.visibility_threshold
        )

        return {
            "torso_visibility": (
                torso_visibility
            ),

            "lower_body_visibility": (
                lower_body_visibility
            ),

            "torso_horizontal": bool(
                torso_horizontal
            ),

            "torso_upright": bool(
                torso_upright
            ),
        }

    def _update_body_information(
        self,
        raw_keypoints,
        visibility,
    ):
        body_information = (
            self._get_body_information(
                raw_keypoints,
                visibility,
            )
        )

        self.torso_visibility = (
            body_information[
                "torso_visibility"
            ]
        )

        self.lower_body_visibility = (
            body_information[
                "lower_body_visibility"
            ]
        )

        self.torso_horizontal = (
            body_information[
                "torso_horizontal"
            ]
        )

        self.torso_upright = (
            body_information[
                "torso_upright"
            ]
        )

    def _update_motion(
        self,
        raw_keypoints,
        current_time,
    ):
        if (
            self.previous_raw_keypoints
            is not None
        ):
            difference = (
                raw_keypoints[:, :2]
                - self.previous_raw_keypoints[:, :2]
            )

            motion = float(
                np.mean(
                    np.linalg.norm(
                        difference,
                        axis=1,
                    )
                )
            )

            self.motion_history.append(
                (
                    current_time,
                    motion,
                )
            )

        self.previous_raw_keypoints = (
            raw_keypoints.copy()
        )

        if not self.motion_history:
            self.recent_motion = 0.0
            return

        recent_values = [
            motion
            for recorded_time, motion
            in self.motion_history

            if (
                current_time
                - recorded_time
            ) <= 1.0
        ]

        if recent_values:
            self.recent_motion = float(
                np.mean(
                    recent_values
                )
            )

        else:
            self.recent_motion = 0.0

    # ==================================================
    # LSTM 예측
    # ==================================================

    def predict(self):
        if len(
            self.sequence
        ) < self.seq_len:
            return (
                "waiting",
                0.0,
            )

        sequence_array = np.asarray(
            self.sequence,
            dtype=np.float32,
        )

        expected_shape = (
            self.seq_len,

            self.num_keypoints
            * self.features,
        )

        if (
            sequence_array.shape
            != expected_shape
        ):
            raise ValueError(
                "시퀀스 크기가 올바르지 않습니다. "
                f"현재: {sequence_array.shape}, "
                f"예상: {expected_shape}"
            )

        input_tensor = torch.from_numpy(
            sequence_array
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(
                input_tensor
            )

            probabilities = (
                torch.softmax(
                    output,
                    dim=1,
                )
            )

            (
                confidence,
                label_index,
            ) = torch.max(
                probabilities,
                dim=1,
            )

        index = int(
            label_index.item()
        )

        confidence_value = float(
            confidence.item()
        )

        label = self.label_map.get(
            index,

            f"unknown_{index}",
        )

        return (
            label,
            confidence_value,
        )

    # ==================================================
    # 프레임 처리
    # ==================================================

    def process_frame(
        self,
        frame,
    ):
        rgb_frame = cv2.cvtColor(
            frame,

            cv2.COLOR_BGR2RGB,
        )

        rgb_frame.flags.writeable = (
            False
        )

        results = self.pose.process(
            rgb_frame
        )

        rgb_frame.flags.writeable = (
            True
        )

        now = datetime.now()

        current_time = (
            time.monotonic()
        )

        if (
            results.pose_landmarks
            is not None
        ):
            self.last_pose_time = now

            self.mp_drawing.draw_landmarks(
                frame,

                results.pose_landmarks,

                self.mp_pose.POSE_CONNECTIONS,
            )

            raw_keypoints = (
                self.extract_keypoints(
                    results
                )
            )

            visibility = (
                self.extract_visibility(
                    results
                )
            )

            if (
                raw_keypoints
                is not None

                and visibility
                is not None
            ):
                # 첫 번째 경로:
                # 원본 좌표로 골반 위치, 속도, 가속도 계산

                self._update_hip_kinematics(
                    raw_keypoints,

                    visibility,

                    current_time,
                )

                self._update_body_information(
                    raw_keypoints,

                    visibility,
                )

                self._update_motion(
                    raw_keypoints,

                    current_time,
                )

                # 두 번째 경로:
                # 정규화한 좌표를 LSTM 입력으로 사용

                normalized_keypoints = (
                    self.normalize_keypoints(
                        raw_keypoints
                    )
                )

                self.sequence.append(
                    normalized_keypoints
                )

            if (
                len(
                    self.sequence
                )
                == self.seq_len
            ):
                (
                    label,

                    confidence,
                ) = self.predict()

                self._update_prediction(
                    label,

                    confidence,

                    current_time,
                )

        else:
            self._handle_no_pose(
                now
            )

        self.draw_status(
            frame
        )

        return frame

    # ==================================================
    # 넘어짐 후보 및 최종 판정
    # ==================================================

    def _clear_fall_candidate(self):
        self.fall_candidate_active = (
            False
        )

        self.fall_candidate_start_time = (
            None
        )

    def _update_prediction(
        self,
        label,
        confidence,
        current_time,
    ):
        send_discord_alert = False

        with self.state_lock:
            self.current_label = (
                label
            )

            self.current_confidence = (
                confidence
            )

            model_says_fall = (
                label == "fall"

                and confidence
                >= self.confidence_threshold
            )

            if model_says_fall:
                self.fall_count += 1

            else:
                self.fall_count = 0

            rapid_fall_motion = (
                self._has_recent_rapid_fall_motion(
                    current_time
                )
            )

            visible_body = (
                self.torso_visibility
                >= self.visibility_threshold
            )

            possible_fall_posture = (
                self.torso_horizontal

                or (
                    label == "fall"

                    and not self.torso_upright
                )
            )

            new_fall_candidate = (
                model_says_fall

                and self.fall_count
                >= self.required_fall_frames

                and rapid_fall_motion

                and visible_body

                and possible_fall_posture

                and not self.fall_detected
            )

            if (
                new_fall_candidate

                and not self.fall_candidate_active
            ):
                self.fall_candidate_active = (
                    True
                )

                self.fall_candidate_start_time = (
                    current_time
                )

                print(
                    "[INFO] 넘어짐 후보 감지: "

                    f"confidence={confidence:.4f}, "

                    "velocity="
                    f"{self.vertical_velocity:.4f}, "

                    "acceleration="
                    f"{self.vertical_acceleration:.4f}"
                )

            if not (
                self.fall_candidate_active
            ):
                return

            candidate_duration = (
                current_time

                - self.fall_candidate_start_time
            )

            if (
                candidate_duration
                > self.candidate_timeout
            ):
                self._clear_fall_candidate()

                return

            # 몸통이 다시 똑바로 서 있으면
            # 실제 넘어짐으로 보지 않고 후보를 해제합니다.

            recovered_posture = (
                self.torso_upright

                and label in (
                    "normal",
                    "sitting",
                )
            )

            if recovered_posture:
                self._clear_fall_candidate()

                return

            sustained_fall_posture = (
                self.torso_horizontal

                or label in (
                    "lying",
                    "fall",
                )
            )

            low_motion = (
                len(
                    self.motion_history
                ) > 0

                and self.recent_motion
                < self.motion_threshold
            )

            final_fall_detected = (
                candidate_duration
                >= self.no_motion_seconds

                and sustained_fall_posture

                and low_motion

                and not self.fall_detected
            )

            if final_fall_detected:
                self.fall_detected = (
                    True
                )

                self.detected_at = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                self.event_id += 1

                self._clear_fall_candidate()

                send_discord_alert = (
                    True
                )

                print(
                    "[ALERT] 넘어짐 최종 감지: "

                    f"{self.detected_at}, "

                    f"confidence={confidence:.4f}, "

                    "motion="
                    f"{self.recent_motion:.5f}"
                )

        if send_discord_alert:
            self._send_discord_async()

    # ==================================================
    # 사람이 검출되지 않을 때
    # ==================================================

    def _handle_no_pose(
        self,
        now,
    ):
        if (
            self.last_pose_time
            is None
        ):
            return

        elapsed = (
            now
            - self.last_pose_time
        ).total_seconds()

        if (
            elapsed
            < self.no_pose_reset_seconds
        ):
            return

        self.sequence.clear()

        self.hip_y_history.clear()

        self.motion_history.clear()

        self.previous_raw_keypoints = (
            None
        )

        self.vertical_velocity = (
            0.0
        )

        self.vertical_acceleration = (
            0.0
        )

        self.velocity_event_time = (
            None
        )

        self.acceleration_event_time = (
            None
        )

        self.recent_motion = (
            0.0
        )

        self.torso_horizontal = (
            False
        )

        self.torso_upright = (
            False
        )

        self.torso_visibility = (
            0.0
        )

        self.lower_body_visibility = (
            0.0
        )

        with self.state_lock:
            self.current_label = (
                "no_person"
            )

            self.current_confidence = (
                0.0
            )

            self.fall_count = (
                0
            )

            self._clear_fall_candidate()

    # ==================================================
    # 화면 표시
    # ==================================================

    def draw_status(
        self,
        frame,
    ):
        with self.state_lock:
            label = (
                self.current_label
            )

            confidence = (
                self.current_confidence
            )

            fall_alert = (
                self.fall_detected
            )

            fall_candidate = (
                self.fall_candidate_active
            )

            candidate_start = (
                self.fall_candidate_start_time
            )

        height, width = (
            frame.shape[:2]
        )

        if fall_alert:
            color = (
                0,
                0,
                255,
            )

            text = (
                "FALL ALERT | "

                f"{label.upper()} "

                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,

                (
                    2,
                    2,
                ),

                (
                    width - 3,
                    height - 3,
                ),

                color,

                14,
            )

        elif fall_candidate:
            color = (
                0,
                100,
                255,
            )

            elapsed = (
                time.monotonic()
                - candidate_start

                if candidate_start
                is not None

                else 0.0
            )

            text = (
                "CHECKING FALL "

                f"{elapsed:.1f}/"

                f"{self.no_motion_seconds:.0f}s"
            )

        elif label == "fall":
            color = (
                0,
                100,
                255,
            )

            text = (
                "FALL CANDIDATE "

                f"{confidence:.2f}"
            )

        elif label == "lying":
            color = (
                0,
                165,
                255,
            )

            text = (
                "LYING "

                f"{confidence:.2f}"
            )

        elif label == "sitting":
            color = (
                255,
                255,
                0,
            )

            text = (
                "SITTING "

                f"{confidence:.2f}"
            )

        elif label == "normal":
            color = (
                0,
                255,
                0,
            )

            text = (
                "NORMAL "

                f"{confidence:.2f}"
            )

        elif label == "no_person":
            color = (
                180,
                180,
                180,
            )

            text = (
                "NO PERSON"
            )

        else:
            color = (
                255,
                255,
                255,
            )

            text = (
                "WAITING FOR SEQUENCE"
            )

        cv2.rectangle(
            frame,

            (
                10,
                10,
            ),

            (
                min(
                    width - 10,
                    620,
                ),

                90,
            ),

            (
                0,
                0,
                0,
            ),

            -1,
        )

        cv2.putText(
            frame,

            text,

            (
                20,
                40,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2,

            cv2.LINE_AA,
        )

        metrics_text = (
            "V: "

            f"{self.vertical_velocity:.2f} "

            "| A: "

            f"{self.vertical_acceleration:.2f} "

            "| M: "

            f"{self.recent_motion:.4f}"
        )

        cv2.putText(
            frame,

            metrics_text,

            (
                20,
                72,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                220,
                220,
                220,
            ),

            1,

            cv2.LINE_AA,
        )

    # ==================================================
    # 상태 조회 및 초기화
    # ==================================================

    def get_status(self):
        with self.state_lock:
            if (
                self.fall_candidate_active

                and self.fall_candidate_start_time
                is not None
            ):
                candidate_seconds = (
                    time.monotonic()

                    - self.fall_candidate_start_time
                )

            else:
                candidate_seconds = (
                    0.0
                )

            return {
                "fall_detected": (
                    self.fall_detected
                ),

                "fall_candidate": (
                    self.fall_candidate_active
                ),

                "candidate_seconds": round(
                    candidate_seconds,
                    2,
                ),

                "label": (
                    self.current_label
                ),

                "confidence": round(
                    self.current_confidence,
                    4,
                ),

                "detected_at": (
                    self.detected_at
                ),

                "event_id": (
                    self.event_id
                ),

                "fall_count": (
                    self.fall_count
                ),

                "required_fall_frames": (
                    self.required_fall_frames
                ),

                "vertical_velocity": round(
                    self.vertical_velocity,
                    4,
                ),

                "vertical_acceleration": round(
                    self.vertical_acceleration,
                    4,
                ),

                "recent_motion": round(
                    self.recent_motion,
                    5,
                ),

                "torso_horizontal": (
                    self.torso_horizontal
                ),

                "torso_upright": (
                    self.torso_upright
                ),

                "last_video_name": (
                    self.last_video_name
                ),
            }

    def reset_fall(self):
        """
        넘어짐 경고와 후보 상태를 초기화합니다.

        자세 분류에 사용하는 LSTM 시퀀스는 유지합니다.
        """

        with self.state_lock:
            self.fall_detected = (
                False
            )

            self.detected_at = (
                None
            )

            self.fall_count = (
                0
            )

            self.velocity_event_time = (
                None
            )

            self.acceleration_event_time = (
                None
            )

            self._clear_fall_candidate()

    def set_saved_video(
        self,
        video_path,
    ):
        if video_path is None:
            return

        with self.state_lock:
            self.last_video_path = str(
                video_path
            )

            self.last_video_name = (
                Path(
                    video_path
                ).name
            )

    # ==================================================
    # Discord 알림
    # ==================================================

    def _send_discord_async(self):
        discord_cfg = self.cfg.get(
            "discord",
            {},
        )

        if not discord_cfg.get(
            "enable",
            False,
        ):
            return

        thread = threading.Thread(
            target=self._send_discord,

            daemon=True,
        )

        thread.start()

    def _send_discord(self):
        discord_cfg = self.cfg.get(
            "discord",
            {},
        )

        # 환경변수가 설정되어 있다면 우선 사용합니다.
        webhook_url = os.getenv(
            "DISCORD_WEBHOOK_URL"
        )

        # 환경변수가 없다면 config.yaml의 설정값을 사용합니다.
        if not webhook_url:
            webhook_url = discord_cfg.get(
                "webhook_url",
                "",
            )

        if (
            not webhook_url

            or webhook_url
            == "YOUR_DISCORD_WEBHOOK_URL"
        ):
            print(
                "[WARNING] Discord Webhook URL이 "
                "설정되지 않았습니다."
            )

            return

        with self.state_lock:
            detected_at = (
                self.detected_at
            )

            confidence = (
                self.current_confidence
            )

            label = (
                self.current_label
            )

            velocity = (
                self.vertical_velocity
            )

            acceleration = (
                self.vertical_acceleration
            )

        message = {
            "content": (
                "🚨 넘어짐이 감지되었습니다.\n"

                f"- 감지 시간: {detected_at}\n"

                f"- 현재 자세: {label}\n"

                "- 신뢰도: "
                f"{confidence * 100:.1f}%\n"

                "- 수직 속도: "
                f"{velocity:.3f}\n"

                "- 수직 가속도: "
                f"{acceleration:.3f}"
            )
        }

        try:
            response = requests.post(
                webhook_url,

                json=message,

                timeout=5,
            )

            response.raise_for_status()

            print(
                "[INFO] Discord 알림 전송 완료"
            )

        except requests.RequestException as error:
            print(
                "[WARNING] Discord 알림 실패: "
                f"{error}"
            )

    # ==================================================
    # 종료 처리
    # ==================================================

    def close(self):
        self.pose.close()
