import argparse
import queue
import threading
import time
import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect


class StreamClient:
    """Zero-latency, jitter-free video stream client displaying frames with OpenCV.

    Emulates browser-grade rendering:
    1. Background worker performs non-blocking I/O AND multi-threaded C++ JPEG decoding.
    2. GUI thread is strictly dedicated to rendering pre-decoded frames at steady V-Sync pace.
    3. Drops stale frames so rendering never lags behind the live camera stream.
    """

    def __init__(
        self,
        host: str,
        port: int,
        display_fps: int = 60,
        window_name: str = "Drone Camera",
    ):
        self.uri = f"ws://{host}:{port}"
        self.window_name = window_name
        self.display_fps = display_fps
        self.frame_interval = 1.0 / max(1, display_fps)

        # Thread-safe queue containing only pre-decoded NumPy BGR frames
        self.decoded_frame_queue = queue.Queue(maxsize=1)
        self.running = threading.Event()
        self.recv_fps = 0.0

    def _network_and_decode_worker(self):
        """Worker thread: Receives JPEG bytes and decodes them off the GUI thread."""
        while self.running.is_set():
            print(f"[Client] Connecting to {self.uri} ...")
            try:
                with connect(
                    self.uri, max_size=None, ping_interval=3, ping_timeout=3
                ) as ws:
                    print(f"[Client] Connected to {self.uri}!")
                    recv_count = 0
                    fps_timer = time.time()

                    while self.running.is_set():
                        data = ws.recv()
                        if not isinstance(data, (bytes, bytearray)):
                            continue

                        # Fast JPEG boundary sanity check (prevents corrupt decoder stalls)
                        if len(data) < 4 or not (
                            data.startswith(b"\xff\xd8")
                            and data.endswith(b"\xff\xd9")
                        ):
                            continue

                        # Multi-threaded C++ JPEG decompression (releases Python GIL)
                        np_arr = np.frombuffer(data, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        # Replace older unrendered frame with the newest frame
                        if self.decoded_frame_queue.full():
                            try:
                                self.decoded_frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self.decoded_frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass

                        recv_count += 1
                        elapsed = time.time() - fps_timer
                        if elapsed >= 1.0:
                            self.recv_fps = recv_count / elapsed
                            recv_count = 0
                            fps_timer = time.time()

            except (ConnectionClosed, ConnectionRefusedError, OSError) as e:
                if self.running.is_set():
                    print(
                        f"[Client] Connection lost ({e}). Reconnecting in 1.5s..."
                    )
                    time.sleep(1.5)

    def run(self):
        """Runs the OpenCV GUI loop paced to monitor refresh rate (like requestAnimationFrame)."""
        self.running.set()
        worker = threading.Thread(
            target=self._network_and_decode_worker, daemon=True
        )
        worker.start()

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        print(
            f"[Client] Display active ({self.display_fps} Hz). Press 'q' or click [X] to exit."
        )

        display_count = 0
        display_fps = 0.0
        fps_timer = time.time()
        last_frame = None

        try:
            while self.running.is_set():
                tick_start = time.perf_counter()

                # Get newest pre-decoded frame if available (non-blocking)
                try:
                    new_frame = self.decoded_frame_queue.get_nowait()
                    last_frame = new_frame
                    display_count += 1
                except queue.Empty:
                    pass

                # If we have a frame, render it
                if last_frame is not None:
                    # Shallow copy for text overlay to avoid modifying original array
                    display_image = last_frame.copy()
                    h, w = display_image.shape[:2]

                    elapsed = time.time() - fps_timer
                    if elapsed >= 1.0:
                        display_fps = display_count / elapsed
                        display_count = 0
                        fps_timer = time.time()

                    cv2.putText(
                        display_image,
                        f"Recv: {self.recv_fps:.1f} FPS | Render: {display_fps:.1f} FPS ({w}x{h})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow(self.window_name, display_image)

                # Process OS window manager events (keep GUI responsive)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if (
                    cv2.getWindowProperty(
                        self.window_name, cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    break

                # Frame Pacing: sleep just enough to match target refresh rate
                time_taken = time.perf_counter() - tick_start
                sleep_duration = self.frame_interval - time_taken
                if sleep_duration > 0.001:
                    time.sleep(sleep_duration)

        finally:
            self.running.clear()
            cv2.destroyAllWindows()
            print("[Client] Closed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Speed Drone Video Client (OpenCV)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="172.21.35.248",
        help="Raspberry Pi IP address (default: 172.21.35.248)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--display-fps",
        type=int,
        default=60,
        help="Target monitor refresh/display rate in Hz (default: 60)",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Drone Camera",
        help="OpenCV window title (default: 'Drone Camera')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = StreamClient(
        host=args.host,
        port=args.port,
        display_fps=args.display_fps,
        window_name=args.window_name,
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user.")


if __name__ == "__main__":
    main()
