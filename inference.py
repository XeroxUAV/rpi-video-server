import argparse
import asyncio
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
import cv2
import numpy as np
import onnxruntime as ort


class ONNXDetector:
    """High-performance YOLOv8 ONNX object detector."""

    def __init__(
        self,
        model_path: str = "onnx-models/best_1.onnx",
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        available_providers = ort.get_available_providers()
        if device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device.lower() == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available_providers
                else ["CPUExecutionProvider"]
            )

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 480, 640]
        self.net_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 480
        self.net_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

        # Extract class names from ONNX metadata if present
        self.class_names: Dict[int, str] = {0: "Window-Iran-Open-V1"}
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                names_dict = ast.literal_eval(meta["names"])
                self.class_names = {int(k): str(v) for k, v in names_dict.items()}
        except Exception:
            pass

        print(
            f"[AI Detector] Loaded '{model_path}' on {self.active_provider} "
            f"(Input: {self.net_w}x{self.net_h}, Classes: {self.class_names})"
        )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """Runs inference on a BGR image and returns detected boxes and inference time in ms."""
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]

        # Resize if incoming stream resolution differs from model expected dimensions (640x480)
        if (orig_w, orig_h) != (self.net_w, self.net_h):
            resized = cv2.resize(
                frame_bgr, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR
            )
            scale_x = orig_w / self.net_w
            scale_y = orig_h / self.net_h
        else:
            resized = frame_bgr
            scale_x, scale_y = 1.0, 1.0

        # Preprocessing: BGR -> RGB, normalize 0..1, CHW, batch dim [1, 3, H, W]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(
            (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        )

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: blob})
        output0 = outputs[0]  # Shape [1, 37, 6300]

        preds = output0[0].T  # Shape [6300, 37]
        boxes_xywh = preds[:, :4]  # cx, cy, w, h
        scores = preds[:, 4]  # Class 0 confidence score

        mask = scores > self.conf_thresh
        cand_boxes = boxes_xywh[mask]
        cand_scores = scores[mask]

        detections: List[Dict] = []
        if len(cand_boxes) > 0:
            boxes_for_nms = []
            for (cx, cy, w, h) in cand_boxes:
                x1 = (cx - w / 2.0) * scale_x
                y1 = (cy - h / 2.0) * scale_y
                bw = w * scale_x
                bh = h * scale_y
                boxes_for_nms.append([int(x1), int(y1), int(bw), int(bh)])

            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, cand_scores.tolist(), self.conf_thresh, self.nms_thresh
            )
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, bw, bh = boxes_for_nms[idx]
                    score = float(cand_scores[idx])
                    class_name = self.class_names.get(0, "Object")
                    detections.append(
                        {
                            "box": (max(0, x), max(0, y), bw, bh),
                            "score": score,
                            "class_id": 0,
                            "class_name": class_name,
                        }
                    )

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return detections, inference_ms


class WebRTCInferenceClient:
    """High-speed desktop WebRTC client with real-time, non-blocking ONNX object detection."""

    @staticmethod
    def _parse_endpoint(host: str, port: int) -> Tuple[str, int]:
        """Parses host and port flexibly (e.g. '10.162.37.248:8080' or 'http://10.162.37.248:8080')."""
        if host.startswith("http://"):
            host = host[7:]
        elif host.startswith("https://"):
            host = host[8:]
        host = host.rstrip("/")
        if ":" in host:
            parts = host.split(":", 1)
            host = parts[0]
            try:
                port = int(parts[1].split("/")[0])
            except ValueError:
                pass
        return host, port

    def __init__(
        self,
        host: str,
        port: int,
        display_fps: int = 60,
        window_name: str = "Drone Camera (AI Detection)",
        record: bool | str = False,
        record_fps: float = 30.0,
        detector: Optional[ONNXDetector] = None,
        stun_server: Optional[str] = None,
    ):
        self.host, self.port = self._parse_endpoint(host, port)
        self.offer_url = f"http://{self.host}:{self.port}/offer"
        self.window_name = window_name
        self.display_fps = display_fps
        self.frame_interval = 1.0 / max(1, display_fps)
        self.detector = detector
        self.stun_server = stun_server

        # Status tracking for GUI splash screen
        self.status: str = "Initializing..."
        self.error_msg: Optional[str] = None

        # Queues: maxsize=1 ensures stale frames are dropped, preventing bufferbloat
        self.frame_queue = queue.Queue(maxsize=1)
        self.inference_queue = queue.Queue(maxsize=1)

        self.running = threading.Event()
        self.recv_fps = 0.0
        self.loop = None
        self.pc = None

        # Atomic detection cache (thread-safe sharing between AI worker and GUI)
        self._det_lock = threading.Lock()
        self._latest_detections: List[Dict] = []
        self._inference_fps: float = 0.0
        self._inference_ms: float = 0.0

        # Video recording configuration
        self.record_enabled = bool(record)
        if self.record_enabled:
            if isinstance(record, str):
                self.record_filename = record
            else:
                self.record_filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            self.record_fps = record_fps
            self.record_queue = queue.Queue(maxsize=300)
            self.record_thread = None
            self.video_writer = None
            self.recorded_frames_count = 0
        else:
            self.record_filename = None
            self.record_fps = None
            self.record_queue = None
            self.record_thread = None
            self.video_writer = None
            self.recorded_frames_count = 0

    async def _run_webrtc(self):
        # Configure ICE servers (disabled by default on LAN to eliminate STUN timeouts)
        ice_servers = []
        if self.stun_server:
            ice_servers.append(RTCIceServer(urls=[self.stun_server]))
        config = RTCConfiguration(iceServers=ice_servers)
        self.pc = RTCPeerConnection(configuration=config)

        self._consumer_task = None

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self.status = "Streaming video"
                print("[WebRTC] Video track received! Streaming frames to inference & GUI...", flush=True)
                self._consumer_task = asyncio.ensure_future(self._track_consumer(track))

        @self.pc.on("connectionstatechange")
        def on_connectionstatechange():
            print(f"[WebRTC] Connection state: {self.pc.connectionState}", flush=True)
            if self.pc.connectionState in ["failed", "closed"]:
                if self.status != "Streaming video":
                    self.error_msg = f"WebRTC connection {self.pc.connectionState}"

        # Add receive-only transceiver
        self.pc.addTransceiver("video", direction="recvonly")

        # Create offer
        self.status = "Creating local SDP offer..."
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering with safety timeout (1.5s max)
        self.status = "Gathering ICE candidates..."
        gather_start = time.time()
        while self.pc.iceGatheringState != "complete" and (time.time() - gather_start) < 1.5:
            await asyncio.sleep(0.05)

        # Send offer to server
        self.status = f"Connecting to http://{self.host}:{self.port}..."
        print(f"[WebRTC] Sending SDP offer to {self.offer_url} ...", flush=True)
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.offer_url,
                json={"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Server rejected offer (HTTP {response.status}): {text}")
                answer_data = await response.json()

        # Apply remote answer
        self.status = "Exchanging SDP answer..."
        answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
        await self.pc.setRemoteDescription(answer)
        self.status = "Connected! Awaiting video track..."
        print("[WebRTC] Handshake complete. Live video stream active.", flush=True)

        # Keep WebRTC loop running while self.running is set
        while self.running.is_set():
            await asyncio.sleep(0.1)

        if self._consumer_task is not None:
            self._consumer_task.cancel()
        await self.pc.close()

    async def _track_consumer(self, track):
        """Asynchronously consumes decoded frames from the WebRTC track with zero blocking."""
        recv_count = 0
        fps_timer = time.time()

        while self.running.is_set():
            try:
                frame = await track.recv()
            except Exception as e:
                print(f"[WebRTC] Track closed ({e})")
                break

            # Convert av.VideoFrame to NumPy BGR image
            img = frame.to_ndarray(format="bgr24")

            # 1. Non-blocking dispatch to AI inference worker (drops stale frames if worker is busy)
            if self.detector is not None:
                if self.inference_queue.full():
                    try:
                        self.inference_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.inference_queue.put_nowait(img)
                except queue.Full:
                    pass

            # 2. Non-blocking dispatch to GUI display (drops stale unrendered frames)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(img)
            except queue.Full:
                pass

            # 3. Non-blocking dispatch to recorder queue (if enabled)
            if self.record_enabled and self.record_queue is not None:
                try:
                    self.record_queue.put_nowait(img)
                except queue.Full:
                    pass

            recv_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.recv_fps = recv_count / elapsed
                recv_count = 0
                fps_timer = time.time()

    def _inference_worker(self):
        """Dedicated background thread running ONNX object detection without blocking WebRTC."""
        inf_count = 0
        fps_timer = time.time()

        while self.running.is_set():
            try:
                frame = self.inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.detector is not None:
                try:
                    detections, inf_ms = self.detector.detect(frame)
                    with self._det_lock:
                        self._latest_detections = detections
                        self._inference_ms = inf_ms

                    inf_count += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        with self._det_lock:
                            self._inference_fps = inf_count / elapsed
                        inf_count = 0
                        fps_timer = time.time()
                except Exception as e:
                    print(f"[AI Detector Error] {e}")

    def _record_worker(self):
        """Dedicated background thread to write recorded frames to disk."""
        while self.running.is_set() or (
            self.record_queue is not None and not self.record_queue.empty()
        ):
            try:
                frame = self.record_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.video_writer = cv2.VideoWriter(
                        self.record_filename, fourcc, self.record_fps, (w, h)
                    )
                    if not self.video_writer.isOpened():
                        print(f"[Record Error] Failed to open VideoWriter for '{self.record_filename}'")
                        self.video_writer = None
                        continue
                    print(
                        f"[Record] Started recording to '{self.record_filename}' "
                        f"({w}x{h} @ {self.record_fps:.1f} FPS)"
                    )

                self.video_writer.write(frame)
                self.recorded_frames_count += 1
            except Exception as e:
                print(f"[Record Error] Failed to write frame: {e}")
            finally:
                self.record_queue.task_done()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(
                f"[Record] Finished: saved {self.recorded_frames_count} frames to '{self.record_filename}'"
            )

    def _render_splash(self) -> np.ndarray:
        """Renders an informative placeholder splash screen while connecting or if errors occur."""
        splash = np.zeros((480, 640, 3), dtype=np.uint8)
        splash[:] = (22, 18, 18)

        # Header
        cv2.putText(
            splash,
            "Drone Camera (WebRTC + AI Detection)",
            (35, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 200),
            2,
        )

        # Target endpoint
        cv2.putText(
            splash,
            f"Target Server: http://{self.host}:{self.port}",
            (35, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
        )

        # AI Model info
        model_name = os.path.basename(self.detector.model_path) if self.detector else "None"
        cv2.putText(
            splash,
            f"AI Model: {model_name} (Device: {getattr(self.detector, 'active_provider', 'N/A')})",
            (35, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 160, 160),
            1,
        )

        # Status / Error box
        if self.error_msg:
            cv2.rectangle(splash, (30, 240), (610, 330), (20, 20, 50), -1)
            cv2.rectangle(splash, (30, 240), (610, 330), (0, 0, 220), 1)
            cv2.putText(
                splash,
                f"Connection Error: {self.error_msg[:50]}",
                (45, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (60, 60, 255),
                1,
            )
            cv2.putText(
                splash,
                "Check network connection & ensure server_webrtc.py is running.",
                (45, 305),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
            )
        else:
            dots = "." * (int(time.time() * 3) % 4)
            cv2.rectangle(splash, (30, 240), (610, 330), (35, 30, 25), -1)
            cv2.rectangle(splash, (30, 240), (610, 330), (70, 60, 50), 1)
            cv2.putText(
                splash,
                f"Status: {self.status}{dots}",
                (45, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (50, 220, 255),
                1,
            )
            cv2.putText(
                splash,
                "Establishing peer connection & awaiting video stream...",
                (45, 305),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (160, 160, 160),
                1,
            )

        # Footer instructions
        cv2.putText(
            splash,
            "Press 'q' or close window to exit",
            (35, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (110, 110, 110),
            1,
        )
        return splash

    def _webrtc_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_webrtc())
        except Exception as e:
            self.error_msg = str(e)
            print(f"[WebRTC] Error in network thread: {e}", flush=True)
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.close()
            except Exception:
                pass

    def run(self):
        """Runs the OpenCV GUI loop on the main thread paced to monitor V-Sync."""
        self.running.set()

        # 1. Initialize OpenCV Qt window and show splash screen on main thread FIRST
        # (Must be done before launching background threads to avoid X11/Qt display deadlock)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        initial_splash = self._render_splash()
        cv2.imshow(self.window_name, initial_splash)
        cv2.waitKey(1)

        rec_info = f" [Recording to {self.record_filename}]" if self.record_enabled else ""
        print(
            f"[Client] Display ready ({self.display_fps} Hz).{rec_info} "
            f"Press 'q' or click [X] to exit."
        )

        # 2. Start WebRTC network thread
        net_thread = threading.Thread(target=self._webrtc_thread, daemon=True)
        net_thread.start()

        # 3. Start AI inference worker thread
        inf_thread = None
        if self.detector is not None:
            inf_thread = threading.Thread(target=self._inference_worker, daemon=True)
            inf_thread.start()

        # 4. Start video recorder thread (if enabled)
        if self.record_enabled:
            self.record_thread = threading.Thread(target=self._record_worker, daemon=True)
            self.record_thread.start()

        display_count = 0
        display_fps = 0.0
        fps_timer = time.time()
        last_frame = None
        frames_rendered = 0

        try:
            while self.running.is_set():
                tick_start = time.perf_counter()

                try:
                    new_frame = self.frame_queue.get_nowait()
                    last_frame = new_frame
                    display_count += 1
                except queue.Empty:
                    pass

                if last_frame is not None:
                    display_image = last_frame.copy()
                    h, w = display_image.shape[:2]

                    # Read latest AI detections atomically
                    with self._det_lock:
                        detections = list(self._latest_detections)
                        inf_fps = self._inference_fps
                        inf_ms = self._inference_ms

                    # Draw bounding boxes and labels
                    for det in detections:
                        bx, by, bw, bh = det["box"]
                        score = det["score"]
                        label = f"{det['class_name']} {score * 100:.1f}%"

                        # Draw bounding box
                        cv2.rectangle(
                            display_image,
                            (bx, by),
                            (bx + bw, by + bh),
                            (0, 255, 0),
                            2,
                        )

                        # Draw label badge
                        (tw, th), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            display_image,
                            (bx, max(0, by - th - 6)),
                            (bx + tw + 6, max(0, by)),
                            (0, 255, 0),
                            -1,
                        )
                        cv2.putText(
                            display_image,
                            label,
                            (bx + 3, max(th, by - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                        )

                    # Performance HUD
                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        display_fps = display_count / elapsed
                        display_count = 0
                        fps_timer = time.time()

                    hud_line1 = (
                        f"WebRTC Recv: {self.recv_fps:.1f} FPS | "
                        f"Render: {display_fps:.1f} FPS ({w}x{h})"
                    )
                    if self.record_enabled:
                        hud_line1 += f" | REC: {self.recorded_frames_count}"

                    hud_line2 = (
                        f"AI Inference: {inf_fps:.1f} FPS ({inf_ms:.1f} ms) | "
                        f"Objects: {len(detections)}"
                    )

                    # Background backdrop for HUD
                    cv2.rectangle(display_image, (6, 6), (460, 54), (10, 14, 20), -1)
                    cv2.rectangle(display_image, (6, 6), (460, 54), (60, 70, 80), 1)

                    cv2.putText(
                        display_image,
                        hud_line1,
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 200),
                        1,
                    )
                    cv2.putText(
                        display_image,
                        hud_line2,
                        (12, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (50, 200, 255),
                        1,
                    )

                    # Red recording dot indicator
                    if self.record_enabled:
                        cv2.circle(display_image, (w - 25, 25), 8, (0, 0, 255), -1)

                    cv2.imshow(self.window_name, display_image)
                else:
                    # Show responsive splash screen while connecting
                    splash = self._render_splash()
                    cv2.imshow(self.window_name, splash)

                # Process window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                frames_rendered += 1
                if frames_rendered > 10:
                    try:
                        prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                        if prop < 0:
                            break
                    except (cv2.error, Exception):
                        # Window was closed by user clicking [X]
                        break

                # Frame pacing to match display refresh rate
                time_taken = time.perf_counter() - tick_start
                sleep_duration = self.frame_interval - time_taken
                if sleep_duration > 0.001:
                    time.sleep(sleep_duration)

        finally:
            self.running.clear()
            if self.record_enabled and self.record_thread is not None:
                print("[Record] Finalizing video file...", flush=True)
                self.record_thread.join(timeout=5.0)
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
            print("[Client] Exited cleanly.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WebRTC Drone Video Client with Real-Time ONNX Object Detection"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="10.162.37.248",
        help="Raspberry Pi IP address or IP:PORT (default: 10.162.37.248)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="WebRTC signaling port (default: 8080)",
    )
    parser.add_argument(
        "--stun",
        type=str,
        default=None,
        help="Optional STUN server URL (default: None for fast LAN connection)",
    )
    parser.add_argument(
        "--display-fps",
        type=int,
        default=60,
        help="Target monitor refresh rate (default: 60)",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Drone Camera (AI Detection)",
        help="OpenCV window title",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="onnx-models/best_1.onnx",
        help="Path to ONNX detection model (default: onnx-models/best_1.onnx)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,
        help="Non-maximum suppression threshold (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference execution device: auto, cuda, or cpu (default: auto)",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=True,
        default=False,
        help="Record video stream to MP4 file (optionally provide filename, default: recording_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=None,
        help="Framerate for recorded video file (default: matches --display-fps or 30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    record_fps = args.record_fps if args.record_fps is not None else float(args.display_fps)

    detector = ONNXDetector(
        model_path=args.model,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        device=args.device,
    )

    client = WebRTCInferenceClient(
        host=args.host,
        port=args.port,
        display_fps=args.display_fps,
        window_name=args.window_name,
        record=args.record,
        record_fps=record_fps,
        detector=detector,
        stun_server=args.stun,
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user.")


if __name__ == "__main__":
    main()
