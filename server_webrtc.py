import argparse
import asyncio
import fractions
import json
import os
import threading
import time
from aiohttp import web
import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender

# Resolution mapping: vertical resolution -> (width, height)
RESOLUTION_MAP = {
    480: (640, 480),
    720: (1280, 720),
    1080: (1920, 1080),
}


class CameraHub:
    """Dedicated background thread for Picamera2 acquisition."""

    def __init__(self, fps: int, resolution: int):
        self.fps = fps
        self.resolution = resolution
        self.width, self.height = RESOLUTION_MAP[resolution]
        self.frame_duration_us = int(1_000_000 / fps)

        self._latest_frame = None
        self._lock = threading.Lock()
        self.running = threading.Event()
        self.capture_thread = None
        self.picam2 = None
        self.active_peers_count = 0

    def start(self):
        try:
            import picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 is not installed. Run this server on your Raspberry Pi where picamera2 is installed."
            ) from e

        self.picam2 = picamera2.Picamera2()

        if self.fps == 90 and self.resolution == 1080:
            print("[Warning] 1080p @ 90 FPS exceeds Pi camera sensor capabilities.")
            print("[Warning] Sensor will operate at its maximum physical limit (~30-50 FPS).")

        sensor_config = None
        if self.fps == 90:
            for mode in self.picam2.sensor_modes:
                if mode.get("fps", 0) >= 90:
                    sensor_config = {
                        "output_size": mode["size"],
                        "bit_depth": mode["bit_depth"],
                    }
                    print(
                        f"[Camera] Selected 90 FPS sensor mode: {mode['size']} @ {mode['fps']:.1f} fps"
                    )
                    break

        config_kwargs = {
            "main": {"size": (self.width, self.height), "format": "BGR888"},
            "controls": {"FrameDurationLimits": (self.frame_duration_us, self.frame_duration_us)},
            "buffer_count": 6 if self.fps == 90 else 4,
        }
        if sensor_config:
            config_kwargs["sensor"] = sensor_config

        config = self.picam2.create_video_configuration(**config_kwargs)
        self.picam2.configure(config)
        self.picam2.start()

        # Enforce FrameDurationLimits after camera startup
        self.picam2.set_controls(
            {"FrameDurationLimits": (self.frame_duration_us, self.frame_duration_us)}
        )
        time.sleep(0.5)

        self.running.set()
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
        print(f"[Camera] Started hardware capture: {self.width}x{self.height} @ {self.fps} FPS")

    def _capture_worker(self):
        frame_count = 0
        fps_start = time.time()

        while self.running.is_set():
            # If no WebRTC peers are currently connected, idle slightly to save CPU
            if self.active_peers_count == 0:
                time.sleep(0.04)

            # Direct BGR capture from ISP (C++ releases Python GIL)
            frame = self.picam2.capture_array()
            if frame is None:
                continue

            with self._lock:
                self._latest_frame = frame

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                if self.active_peers_count > 0:
                    print(
                        f"[Camera Hardware] Capture: {fps:.1f} FPS | Active Peers: {self.active_peers_count}"
                    )
                frame_count = 0
                fps_start = time.time()

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame

    def stop(self):
        self.running.clear()
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        print("[Camera] Stopped hardware capture.")


class CameraStreamTrack(VideoStreamTrack):
    """WebRTC VideoStreamTrack delivering frames from the CameraHub."""

    kind = "video"

    def __init__(self, camera_hub: CameraHub, fps: int = 90):
        super().__init__()
        self.camera_hub = camera_hub
        self.fps = fps
        self.time_base = fractions.Fraction(1, 90000)
        self.frame_duration = 90000 // fps
        self._timestamp = 0
        self._start_time = None

    async def recv(self):
        if self._start_time is None:
            self._start_time = time.time()
            self._timestamp = 0
        else:
            self._timestamp += self.frame_duration
            target_time = self._start_time + (self._timestamp / 90000)
            wait = target_time - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        frame = self.camera_hub.get_latest_frame()
        if frame is None:
            # Generate black placeholder frame during camera warmup
            frame = np.zeros(
                (self.camera_hub.height, self.camera_hub.width, 3), dtype=np.uint8
            )

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._timestamp
        video_frame.time_base = self.time_base
        return video_frame


class WebRTCServer:
    def __init__(self, host: str, port: int, fps: int, resolution: int):
        self.host = host
        self.port = port
        self.fps = fps
        self.resolution = resolution
        self.camera_hub = CameraHub(fps=fps, resolution=resolution)
        self.pcs = set()
        self.app = web.Application()

    async def index(self, request):
        html_path = os.path.join(os.path.dirname(__file__), "web", "webrtc.html")
        if not os.path.exists(html_path):
            return web.Response(
                text="<h1>web/webrtc.html not found</h1>", content_type="text/html"
            )
        return web.FileResponse(html_path)

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)
        self.camera_hub.active_peers_count += 1
        client_ip = request.remote
        print(f"[WebRTC] New peer connection request from {client_ip}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[WebRTC] Peer {client_ip} state -> {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.discard(pc)
                self.camera_hub.active_peers_count = max(
                    0, self.camera_hub.active_peers_count - 1
                )

        # Attach custom camera track to peer connection
        track = CameraStreamTrack(self.camera_hub, fps=self.fps)
        pc.addTrack(track)

        # Prefer H.264 or VP8 codec
        transceiver = pc.getTransceivers()[0]
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = []
        for codec in capabilities.codecs:
            if codec.mimeType.lower() in ["video/h264", "video/vp8"]:
                preferences.append(codec)
        if preferences:
            transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        response_data = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
        return web.json_response(
            response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    async def options_handler(self, request):
        return web.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    async def cleanup(self, app):
        print("\n[WebRTC] Closing all active peer connections...")
        coros = [pc.close() for pc in list(self.pcs)]
        await asyncio.gather(*coros, return_exceptions=True)
        self.pcs.clear()
        self.camera_hub.stop()

    def run(self):
        self.camera_hub.start()

        self.app.router.add_get("/", self.index)
        self.app.router.add_get("/webrtc.html", self.index)
        self.app.router.add_post("/offer", self.offer)
        self.app.router.add_options("/offer", self.options_handler)
        self.app.on_shutdown.append(self.cleanup)

        print(f"[WebRTC] Server listening on http://{self.host}:{self.port}")
        print(f"[WebRTC] Open http://{self.host}:{self.port} in your browser to view the stream.")
        web.run_app(self.app, host=self.host, port=self.port, access_log=None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi WebRTC Camera Broadcasting Server"
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
        help="Vertical resolution: 480 (640x480), 720 (1280x720), or 1080 (1920x1080) (default: 480)",
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
        default=8080,
        help="HTTP/WebRTC signaling port (default: 8080)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server = WebRTCServer(
        host=args.host,
        port=args.port,
        fps=args.fps,
        resolution=args.resolution,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user.")


if __name__ == "__main__":
    main()
