"""
Shared HandTracker utility using MediaPipe Hands.
Provides hand landmark detection, fingertip positions, finger states, and gesture recognition.
"""
import time
import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Wraps MediaPipe Hands for easy hand tracking in games."""

    # Landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    THUMB_IP = 3
    INDEX_DIP = 7
    MIDDLE_DIP = 11
    RING_DIP = 15
    PINKY_DIP = 19

    THUMB_MCP = 2
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    FINGER_DIP_IDS = [3, 7, 11, 15, 19]
    FINGER_MCP_IDS = [2, 5, 9, 13, 17]
    SMOOTHING_ALPHA = 0.42

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.6):
        self.enabled = False
        self.mp_hands = None
        self.mp_draw = None
        self.mp_draw_styles = None
        self.hands = None

        solutions_api = getattr(mp, "solutions", None)
        hands_api = getattr(solutions_api, "hands", None) if solutions_api is not None else None
        draw_api = getattr(solutions_api, "drawing_utils", None) if solutions_api is not None else None
        draw_styles_api = getattr(solutions_api, "drawing_styles", None) if solutions_api is not None else None

        if hands_api is not None and draw_api is not None and draw_styles_api is not None:
            self.mp_hands = hands_api
            self.mp_draw = draw_api
            self.mp_draw_styles = draw_styles_api

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.enabled = True
        else:
            print(
                "[HandTracker] MediaPipe Hands API unavailable in this environment. "
                "Falling back to non-hand controls. Install a compatible mediapipe version."
            )

        self.landmarks = None
        self.hand_detected = False
        self.frame_shape = None
        self._smoothed_landmark_positions = {}

    def process_frame(self, frame):
        """Process a BGR frame and extract hand landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            True if a hand was detected, False otherwise
        """
        if not self.enabled:
            self.hand_detected = False
            self.landmarks = None
            return False

        self.frame_shape = frame.shape[:2]  # (h, w)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            self.landmarks = results.multi_hand_landmarks[0]
            self.hand_detected = True
        else:
            self.landmarks = None
            self.hand_detected = False
            self._smoothed_landmark_positions.clear()

        return self.hand_detected

    def get_landmark_pos(self, landmark_id, frame_w=None, frame_h=None, smooth=True):
        """Get pixel position of a specific landmark.
        
        Args:
            landmark_id: MediaPipe landmark index (0-20)
            frame_w: Target width for coordinate scaling (default: source frame width)
            frame_h: Target height for coordinate scaling (default: source frame height)
            
        Returns:
            (x, y) tuple in pixel coordinates, or None if no hand detected
        """
        if not self.hand_detected or self.landmarks is None:
            return None

        h_src, w_src = self.frame_shape
        w = frame_w if frame_w else w_src
        h = frame_h if frame_h else h_src

        lm = self.landmarks.landmark[landmark_id]
        raw_pos = (int(lm.x * w), int(lm.y * h))

        if not smooth:
            return raw_pos

        key = (landmark_id, w, h)
        prev = self._smoothed_landmark_positions.get(key)
        if prev is None:
            self._smoothed_landmark_positions[key] = raw_pos
            return raw_pos

        alpha = self.SMOOTHING_ALPHA
        smoothed = (
            int(prev[0] * (1.0 - alpha) + raw_pos[0] * alpha),
            int(prev[1] * (1.0 - alpha) + raw_pos[1] * alpha),
        )
        self._smoothed_landmark_positions[key] = smoothed
        return smoothed

    def get_fingertip_pos(self, finger="index", frame_w=None, frame_h=None):
        """Get fingertip position by name.
        
        Args:
            finger: One of 'thumb', 'index', 'middle', 'ring', 'pinky'
            frame_w, frame_h: Optional target dimensions
            
        Returns:
            (x, y) tuple or None
        """
        finger_map = {
            "thumb": self.THUMB_TIP,
            "index": self.INDEX_TIP,
            "middle": self.MIDDLE_TIP,
            "ring": self.RING_TIP,
            "pinky": self.PINKY_TIP,
        }
        landmark_id = finger_map.get(finger, self.INDEX_TIP)
        return self.get_landmark_pos(landmark_id, frame_w, frame_h)

    def get_all_landmarks(self, frame_w=None, frame_h=None):
        """Get all 21 landmarks as a list of (x, y) tuples.
        
        Returns:
            List of 21 (x, y) tuples, or empty list if no hand detected
        """
        if not self.hand_detected or self.landmarks is None:
            return []

        h_src, w_src = self.frame_shape
        w = frame_w if frame_w else w_src
        h = frame_h if frame_h else h_src

        return [
            (int(lm.x * w), int(lm.y * h))
            for lm in self.landmarks.landmark
        ]

    def get_landmark_array(self):
        """Get raw landmark coordinates as a flat numpy array (for ML input).
        
        Returns:
            numpy array of shape (63,) with [x0, y0, z0, x1, y1, z1, ...], or None
        """
        if not self.hand_detected or self.landmarks is None:
            return None

        coords = []
        for lm in self.landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def get_finger_states(self):
        """Determine which fingers are extended (up).
        
        Returns:
            dict with keys 'thumb', 'index', 'middle', 'ring', 'pinky'
            and boolean values (True = extended), or None if no hand
        """
        if not self.hand_detected or self.landmarks is None:
            return None

        lm = self.landmarks.landmark

        # Thumb: compare x of tip vs IP joint (works for right hand, approximate)
        thumb_up = lm[self.THUMB_TIP].x < lm[self.THUMB_IP].x

        # Other fingers: tip is above (lower y) DIP joint
        index_up = lm[self.INDEX_TIP].y < lm[self.INDEX_DIP].y
        middle_up = lm[self.MIDDLE_TIP].y < lm[self.MIDDLE_DIP].y
        ring_up = lm[self.RING_TIP].y < lm[self.RING_DIP].y
        pinky_up = lm[self.PINKY_TIP].y < lm[self.PINKY_DIP].y

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
        }

    def get_finger_count(self):
        """Count number of extended fingers.
        
        Returns:
            int (0-5) or None if no hand detected
        """
        states = self.get_finger_states()
        if states is None:
            return None
        return sum(states.values())

    def get_gesture(self):
        """Recognize basic gestures from finger states.
        
        Returns:
            str: one of 'fist', 'open', 'pointing', 'peace', 'thumbs_up', 'unknown'
            or None if no hand detected
        """
        states = self.get_finger_states()
        if states is None:
            return None

        t, i, m, r, p = (
            states["thumb"],
            states["index"],
            states["middle"],
            states["ring"],
            states["pinky"],
        )

        count = sum([t, i, m, r, p])

        if count == 0:
            return "fist"
        elif count == 5:
            return "open"
        elif i and not m and not r and not p:
            return "pointing"
        elif i and m and not r and not p:
            return "peace"
        elif t and not i and not m and not r and not p:
            return "thumbs_up"
        else:
            return "unknown"

    def get_pinch_distance(self, frame_w=None, frame_h=None):
        """Get distance between thumb tip and index fingertip (for pinch detection).
        
        Returns:
            float distance in pixels, or None
        """
        thumb = self.get_fingertip_pos("thumb", frame_w, frame_h)
        index = self.get_fingertip_pos("index", frame_w, frame_h)
        if thumb is None or index is None:
            return None
        return np.sqrt((thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2)

    def draw_landmarks(self, frame):
        """Draw hand landmarks on the frame.
        
        Args:
            frame: BGR image to draw on (modified in-place)
            
        Returns:
            frame with landmarks drawn
        """
        if self.enabled and self.hand_detected and self.landmarks is not None:
            self.mp_draw.draw_landmarks(
                frame,
                self.landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw_styles.get_default_hand_landmarks_style(),
                self.mp_draw_styles.get_default_hand_connections_style(),
            )
        return frame

    def release(self):
        """Release MediaPipe resources."""
        if self.hands is not None:
            self.hands.close()

    def calibrate(self, cap, duration_seconds=3) -> dict:
        """Run a short calibration session, return recommended settings.
        Call this before starting a game to adapt to current lighting.
        
        Args:
            cap: OpenCV VideoCapture object (already opened)
            duration_seconds: how long to sample (default 3 seconds)
        
        Returns:
            dict with 'detection_rate' (0.0-1.0) and 'recommended_confidence'
        """
        import time
        import cv2

        detections = 0
        total_frames = 0
        start = time.time()

        while time.time() - start < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            self.process_frame(frame)
            total_frames += 1
            if self.hand_detected:
                detections += 1

        detection_rate = detections / max(total_frames, 1)

        # Lower confidence threshold if detection rate is poor
        if detection_rate > 0.8:
            recommended = 0.75
        elif detection_rate > 0.5:
            recommended = 0.60
        else:
            recommended = 0.45

        return {
            "detection_rate": detection_rate,
            "recommended_confidence": recommended,
            "frames_sampled": total_frames,
        }