import argparse
import queue
import threading
import time
import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect


class StreamClient:
    """Zero-latency video stream client displaying frames with OpenCV."""

    def __init__(self, host: str, port: int, window_name: str = "Drone Camera"):
        self.uri = f"ws://{host}:{port}"
        self.window_name = window_name
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = threading.Event()
        self.recv_fps = 0.0

    def _network_worker(self):
        """Worker thread: continuously receives frames and keeps only the freshest frame."""
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

                        # Discard stale frame if main thread hasn't rendered it yet
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self.frame_queue.put_nowait(data)
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
        """Runs the OpenCV GUI loop on the main thread."""
        self.running.set()
        net_thread = threading.Thread(target=self._network_worker, daemon=True)
        net_thread.start()

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        print(
            f"[Client] Display ready. Press 'q' or click the window [X] to exit."
        )

        display_count = 0
        display_fps = 0.0
        fps_timer = time.time()

        try:
            while self.running.is_set():
                try:
                    data = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    # Keep OpenCV window responsive even when no frames arrive
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                display_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    display_fps = display_count / elapsed
                    display_count = 0
                    fps_timer = time.time()

                h, w = frame.shape[:2]
                cv2.putText(
                    frame,
                    f"Recv: {self.recv_fps:.1f} FPS | Disp: {display_fps:.1f} FPS ({w}x{h})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Exit if the window was closed via the 'X' button
                if (
                    cv2.getWindowProperty(
                        self.window_name, cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    break
        finally:
            self.running.clear()
            cv2.destroyAllWindows()
            print("[Client] Closed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drone Camera Stream Client (OpenCV)"
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
        "--window-name",
        type=str,
        default="Drone Camera",
        help="OpenCV window title (default: 'Drone Camera')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = StreamClient(
        host=args.host, port=args.port, window_name=args.window_name
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user.")


if __name__ == "__main__":
    main()
