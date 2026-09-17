import argparse
import asyncio
import queue
import threading
import time
import aiohttp
import cv2
import numpy as np
from aiortc import RTCIceServer, RTCConfiguration, RTCPeerConnection, RTCSessionDescription

import onnxruntime as ort


class WebRTCClient:
    """High-speed desktop OpenCV client receiving video via WebRTC (UDP/SRTP)."""

    def __init__(
        self,
        host: str,
        port: int,
        display_fps: int = 60,
        window_name: str = "Drone Camera (WebRTC)",
        record: bool | str = False,
        record_fps: float = 30.0,
        model_path: str = "onnx-models/best_1.onnx",
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
    ):
        if ":" in host:
            h_part, p_part = host.split(":", 1)
            host = h_part
            try:
                port = int(p_part.split("/")[0])
            except ValueError:
                pass
        self.offer_url = f"http://{host}:{port}/offer"
        self.window_name = window_name
        self.display_fps = display_fps
        self.frame_interval = 1.0 / max(1, display_fps)

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = threading.Event()
        self.recv_fps = 0.0
        self.loop = None
        self.pc = None
        
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.net_h = input_shape[2] if isinstance(input_shape[2], int) else 480
        self.net_w = input_shape[3] if isinstance(input_shape[3], int) else 640
        self.conf_thresh = 0.25
        self.nms_thresh = 0.45
        self.class_names = {0: "Window-Iran-Open-V1"}
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import ast
                names_dict = ast.literal_eval(meta["names"])
                self.class_names = {int(k): str(v) for k, v in names_dict.items()}
        except Exception:
            pass
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
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        self.pc = RTCPeerConnection(configuration=config)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print("[WebRTC] Video track received! Streaming to OpenCV...")
                asyncio.ensure_future(self._track_consumer(track))

        @self.pc.on("connectionstatechange")
        def on_connectionstatechange():
            print(f"[WebRTC] Connection state: {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed"]:
                self.running.clear()

        # Add receive-only transceiver
        self.pc.addTransceiver("video", direction="recvonly")

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Vanilla ICE: Wait until ICE gathering finishes (with safety timeout)
        gather_start = time.time()
        while self.pc.iceGatheringState != "complete" and (time.time() - gather_start) < 1.0:
            await asyncio.sleep(0.05)

        # Send offer to server
        print(f"[WebRTC] Sending SDP offer to {self.offer_url} ...")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.offer_url,
                json={"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Server rejected offer (HTTP {response.status}): {text}")
                answer_data = await response.json()

        # Apply remote answer
        answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
        await self.pc.setRemoteDescription(answer)
        print("[WebRTC] Handshake complete. Stream active.")

        # Keep WebRTC loop running while self.running is set
        while self.running.is_set():
            await asyncio.sleep(0.5)

        await self.pc.close()

    async def _track_consumer(self, track):
        """Asynchronously consumes decoded frames from the WebRTC track."""
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

            # Queue clean frame for recording if enabled
            if self.record_enabled and self.record_queue is not None:
                try:
                    self.record_queue.put_nowait(img)
                except queue.Full:
                    pass

            # Store only the freshest frame (drop older unrendered frames)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(img)
            except queue.Full:
                pass

            recv_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.recv_fps = recv_count / elapsed
                recv_count = 0
                fps_timer = time.time()

    def _record_worker(self):
        """Dedicated background thread to write recorded frames to disk without blocking WebRTC."""
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
                        print(
                            f"[Record Error] Failed to open VideoWriter for '{self.record_filename}'"
                        )
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

    def _webrtc_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_webrtc())
        except Exception as e:
            print(f"[WebRTC] Error in network thread: {e}")
            self.running.clear()
        finally:
            self.loop.close()

    def run(self):
        """Runs the OpenCV GUI loop on the main thread."""
        self.running.set()
        net_thread = threading.Thread(target=self._webrtc_thread, daemon=True)
        net_thread.start()

        if self.record_enabled:
            self.record_thread = threading.Thread(target=self._record_worker, daemon=True)
            self.record_thread.start()

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        rec_info = f" [Recording to {self.record_filename}]" if self.record_enabled else ""
        print(f"[Client] Display ready ({self.display_fps} Hz).{rec_info} Press 'q' or click [X] to exit.")

        display_count = 0
        display_fps = 0.0
        fps_timer = time.time()
        last_frame = None

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

                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        display_fps = display_count / elapsed
                        display_count = 0
                        fps_timer = time.time()

                    hud_text = f"WebRTC Recv: {self.recv_fps:.1f} FPS | Render: {display_fps:.1f} FPS ({w}x{h})"
                    if self.record_enabled:
                        hud_text += f" | REC: {self.recorded_frames_count}"

                    cv2.putText(
                        display_image,
                        hud_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    if self.record_enabled:
                        cv2.circle(display_image, (w - 25, 25), 8, (0, 0, 255), -1)

                    # Preprocessing for ONNX model (640x480, RGB, float32 0..1, NCHW)
                    if (w, h) != (self.net_w, self.net_h):
                        resized = cv2.resize(
                            display_image, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR
                        )
                        scale_x = w / self.net_w
                        scale_y = h / self.net_h
                    else:
                        resized = display_image
                        scale_x, scale_y = 1.0, 1.0

                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    blob = np.ascontiguousarray(
                        (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
                    )

                    outputs = self.session.run(None, {self.input_name: blob})
                    output0 = outputs[0]  # Shape [1, 37, 6300]
                    preds = output0[0].T  # Shape [6300, 37]

                    boxes_xywh = preds[:, :4]
                    scores = preds[:, 4]  # Class 0 confidence score

                    mask = scores > self.conf_thresh
                    cand_boxes = boxes_xywh[mask]
                    cand_scores = scores[mask]

                    if len(cand_boxes) > 0:
                        boxes_for_nms = []
                        for (cx, cy, bw_box, bh_box) in cand_boxes:
                            x1 = int((cx - bw_box / 2.0) * scale_x)
                            y1 = int((cy - bh_box / 2.0) * scale_y)
                            bw = int(bw_box * scale_x)
                            bh = int(bh_box * scale_y)
                            boxes_for_nms.append([x1, y1, bw, bh])

                        indices = cv2.dnn.NMSBoxes(
                            boxes_for_nms, cand_scores.tolist(), self.conf_thresh, self.nms_thresh
                        )
                        if len(indices) > 0:
                            for idx in indices.flatten():
                                bx, by, bw, bh = boxes_for_nms[idx]
                                score = float(cand_scores[idx])
                                class_name = self.class_names.get(0, "Window-Iran-Open-V1")
                                label = f"{class_name} {score * 100:.1f}%"

                                # Draw bounding box
                                cv2.rectangle(
                                    display_image,
                                    (max(0, bx), max(0, by)),
                                    (min(w - 1, bx + bw), min(h - 1, by + bh)),
                                    (0, 255, 0),
                                    2,
                                )

                                # Draw label badge
                                (tw, th), baseline = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                                )
                                cv2.rectangle(
                                    display_image,
                                    (max(0, bx), max(0, by - th - 6)),
                                    (min(w - 1, bx + tw + 6), max(0, by)),
                                    (0, 255, 0),
                                    -1,
                                )
                                cv2.putText(
                                    display_image,
                                    label,
                                    (max(0, bx) + 3, max(th, by - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    1,
                                )

                    cv2.imshow(self.window_name, display_image)

                # Process window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Frame pacing to monitor rate
                time_taken = time.perf_counter() - tick_start
                sleep_duration = self.frame_interval - time_taken
                if sleep_duration > 0.001:
                    time.sleep(sleep_duration)

        finally:
            self.running.clear()
            if self.record_enabled and self.record_thread is not None:
                print("[Record] Finalizing video file...")
                self.record_thread.join(timeout=5.0)
            cv2.destroyAllWindows()
            print("[Client] Exited cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(description="WebRTC Drone Video Client (OpenCV)")
    parser.add_argument(
        "--host",
        type=str,
        default="10.162.37.248",
        help="Raspberry Pi IP address (default: 172.21.35.248)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="WebRTC signaling port (default: 8080)",
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
        default="Drone Camera (WebRTC)",
        help="OpenCV window title",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=True,
        default=False,
        help="Record incoming video stream to file (optionally specify filename, default: recording_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=None,
        help="Framerate for recorded video file (default: matches --display-fps or 30)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="onnx-models/best_1.onnx",
        help="Path to ONNX model (default: onnx-models/best_1.onnx)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.50,
        help="Detection confidence threshold (default: 0.25)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    record_fps = args.record_fps if args.record_fps is not None else float(args.display_fps)
    client = WebRTCClient(
        host=args.host,
        port=args.port,
        display_fps=args.display_fps,
        window_name=args.window_name,
        record=args.record,
        record_fps=record_fps,
        model_path=args.model,
        conf_thresh=args.conf_thresh,
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user.")


if __name__ == "__main__":
    main()

