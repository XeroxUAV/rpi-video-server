import argparse
import asyncio
import threading
import time
import cv2
import websockets

# Mapping vertical resolution argument to (width, height)
RESOLUTION_MAP = {
    480: (640, 480),
    720: (1280, 720),
    1080: (1920, 1080),
}


class CameraServer:
    """High-performance Picamera2 streaming server with zero-latency frame dropping."""

    def __init__(self, fps: int, resolution: int, host: str, port: int, quality: int):
        self.fps = fps
        self.resolution = resolution
        self.width, self.height = RESOLUTION_MAP[resolution]
        self.host = host
        self.port = port
        self.quality = quality

        self.active_client_queues = set()
        self.running = threading.Event()
        self.capture_thread = None
        self.loop = None
        self.picam2 = None

    def start_camera(self):
        """Initializes and configures the camera hardware."""
        try:
            import picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 is not installed. Please run this script on your Raspberry Pi where picamera2 is installed."
            ) from e

        self.picam2 = picamera2.Picamera2()

        # Frame duration in microseconds (e.g. 11111 us for 90 FPS, 33333 us for 30 FPS)
        frame_duration_us = int(1_000_000 / self.fps)

        if self.fps == 90 and self.resolution == 1080:
            print("[Warning] 1080p @ 90 FPS is not supported by Pi camera sensors.")
            print("[Warning] Sensor will operate at its maximum physical framerate (~30-50 FPS).")

        sensor_config = None
        if self.fps == 90:
            # Look for a native sensor mode supporting >= 90 FPS
            for mode in self.picam2.sensor_modes:
                if mode.get("fps", 0) >= 90:
                    sensor_config = {
                        "output_size": mode["size"],
                        "bit_depth": mode["bit_depth"],
                    }
                    print(
                        f"Selected high-speed sensor mode: {mode['size']} @ {mode['fps']:.1f} fps"
                    )
                    break

        config_kwargs = {
            "main": {"size": (self.width, self.height), "format": "BGR888"},
            "controls": {"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            "buffer_count": 6 if self.fps == 90 else 4,
        }
        if sensor_config:
            config_kwargs["sensor"] = sensor_config

        config = self.picam2.create_video_configuration(**config_kwargs)
        self.picam2.configure(config)
        self.picam2.start()

        # Re-enforce FrameDurationLimits after start
        self.picam2.set_controls(
            {"FrameDurationLimits": (frame_duration_us, frame_duration_us)}
        )
        time.sleep(0.5)
        print(f"Camera started: {self.width}x{self.height} @ target {self.fps} FPS")

    def _broadcast_to_queues(self, jpeg_bytes: bytes):
        """Dispatches the freshest frame to each client queue, dropping older frames."""
        for q in list(self.active_client_queues):
            if q.full():
                try:
                    q.get_nowait()  # Drop stale frame to prevent TCP bufferbloat
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(jpeg_bytes)
            except asyncio.QueueFull:
                pass

    def _capture_worker(self):
        """Dedicated thread: captures frames from hardware ISP and encodes JPEGs in C++."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        frame_count = 0
        fps_start = time.time()

        while self.running.is_set():
            # If no clients are connected, sleep briefly to avoid burning CPU
            if not self.active_client_queues:
                time.sleep(0.02)
                continue

            # Direct BGR capture from ISP (C++ releases GIL)
            frame = self.picam2.capture_array()
            if frame is None:
                continue

            # Fast JPEG encode (C++ libjpeg releases GIL)
            success, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            jpeg_bytes = buffer.tobytes()

            # Dispatch frame to asyncio event loop
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self._broadcast_to_queues, jpeg_bytes)

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(
                    f"[Camera Hardware] Capture: {fps:.1f} FPS | Active Clients: {len(self.active_client_queues)}"
                )
                frame_count = 0
                fps_start = time.time()

    async def handle_client(self, websocket):
        """Streams frames to an individual client with zero-lag guarantee."""
        queue = asyncio.Queue(maxsize=1)
        self.active_client_queues.add(queue)
        client_addr = websocket.remote_address
        print(f"[WebSocket] Client connected: {client_addr}")

        try:
            while True:
                # Always wait for the newest frame
                frame_bytes = await queue.get()
                await websocket.send(frame_bytes)
        except websockets.exceptions.ConnectionClosed:
            print(f"[WebSocket] Client disconnected: {client_addr}")
        finally:
            self.active_client_queues.discard(queue)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.start_camera()

        self.running.set()
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=None,
            ping_interval=3,
            ping_timeout=3,
        ):
            print(f"[WebSocket] Server listening on ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass
            finally:
                self.running.clear()
                if self.picam2:
                    self.picam2.stop()
                    self.picam2.close()
                print("Server stopped cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Performance Raspberry Pi Camera WebSocket Server"
    )
    parser.add_argument(
        "--fps",
        type=int,
        choices=[30, 90],
        default=90,
        help="Target framerate: 30 or 90 FPS (default: 90)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[480, 720, 1080],
        default=480,
        help="Resolution height: 480 (640x480), 720 (1280x720), or 1080 (1920x1080) (default: 480)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host IP to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=50,
        help="JPEG quality 1-100 (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server = CameraServer(
        fps=args.fps,
        resolution=args.resolution,
        host=args.host,
        port=args.port,
        quality=args.quality,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer terminated by user.")


if __name__ == "__main__":
    main()
