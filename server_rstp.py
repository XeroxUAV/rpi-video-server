import argparse
import asyncio
import base64
import fractions
import os
import re
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

from aiohttp import web
import av
import cv2
import numpy as np

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

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.running = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.picam2 = None
        self.active_clients_count = 0
        self._use_mock = False

    def start(self):
        try:
            import picamera2
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

            self.picam2.set_controls(
                {"FrameDurationLimits": (self.frame_duration_us, self.frame_duration_us)}
            )
            time.sleep(0.5)
            print(f"[Camera] Started hardware capture: {self.width}x{self.height} @ {self.fps} FPS")

        except (ImportError, Exception) as e:
            if os.environ.get("MOCK_CAMERA", "0") == "1" or isinstance(e, ImportError):
                print(f"[Camera] Picamera2 unavailable ({e}). Using synthetic test pattern generator.")
                self._use_mock = True
            else:
                raise RuntimeError(
                    "picamera2 is not installed or camera unavailable. Run this server on your Raspberry Pi."
                ) from e

        self.running.set()
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def _generate_synthetic_frame(self, frame_id: int) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :, 0] = np.linspace(20, 60, self.width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(30, 80, self.height, dtype=np.uint8)[:, None]
        frame[:, :, 2] = 40

        # Moving ball
        bx = int((np.sin(frame_id * 0.05) + 1.0) * 0.5 * (self.width - 100)) + 50
        by = int((np.cos(frame_id * 0.05) + 1.0) * 0.5 * (self.height - 100)) + 50
        cv2.circle(frame, (bx, by), 30, (0, 255, 128), -1)

        # Time & Info text
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"RTSP Live - {self.width}x{self.height} @ {self.fps} FPS", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {ts} | Frame: {frame_id}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return frame

    def _capture_worker(self):
        frame_count = 0
        fps_start = time.time()
        interval = 1.0 / self.fps

        while self.running.is_set():
            loop_start = time.perf_counter()

            # Idle slightly if no clients to save CPU
            if self.active_clients_count == 0:
                time.sleep(0.04)

            if self._use_mock:
                frame = self._generate_synthetic_frame(frame_count)
            else:
                frame = self.picam2.capture_array()

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                if self.active_clients_count > 0:
                    print(
                        f"[Camera Hardware] Capture: {fps:.1f} FPS | Active Clients: {self.active_clients_count}"
                    )
                frame_count = 0
                fps_start = time.time()

            if self._use_mock:
                work_time = time.perf_counter() - loop_start
                sleep_dur = interval - work_time
                if sleep_dur > 0.001:
                    time.sleep(sleep_dur)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def stop(self):
        self.running.clear()
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        print("[Camera] Stopped hardware capture.")


class H264Encoder:
    """In-memory low-latency H.264 encoder using PyAV."""

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps

        self.codec = av.CodecContext.create("libx264", "w")
        self.codec.width = width
        self.codec.height = height
        self.codec.pix_fmt = "yuv420p"
        self.codec.time_base = fractions.Fraction(1, 90000)
        self.codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "repeat-headers": "1",
        }
        self.codec.gop_size = fps  # 1 keyframe per second
        self.codec.max_b_frames = 0
        self.codec.open()

        self.sps: Optional[bytes] = None
        self.pps: Optional[bytes] = None

    @staticmethod
    def extract_nal_units(data: bytes) -> List[bytes]:
        """Extracts Annex-B NAL units delimited by 0x000001 or 0x00000001."""
        nalus = []
        indices = [m.start() for m in re.finditer(b"\x00\x00\x01|\x00\x00\x00\x01", data)]
        indices.append(len(data))
        for i in range(len(indices) - 1):
            start = indices[i]
            start += 4 if data[start:start+4] == b"\x00\x00\x00\x01" else 3
            end = indices[i + 1]
            nalu = data[start:end]
            if nalu:
                nalus.append(nalu)
        return nalus

    def encode(self, frame_bgr: np.ndarray, pts: int) -> List[bytes]:
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        video_frame = video_frame.reformat(format="yuv420p")
        video_frame.pts = pts

        packets = self.codec.encode(video_frame)
        raw_bytes = b"".join([bytes(p) for p in packets])
        nalus = self.extract_nal_units(raw_bytes)

        # Cache SPS (type 7) and PPS (type 8)
        for n in nalus:
            nal_type = n[0] & 0x1F
            if nal_type == 7 and self.sps is None:
                self.sps = n
            elif nal_type == 8 and self.pps is None:
                self.pps = n

        return nalus

    def get_sprop_parameter_sets(self) -> str:
        """Returns base64 encoded SPS/PPS string for SDP declaration."""
        if self.sps and self.pps:
            sps_b64 = base64.b64encode(self.sps).decode("ascii")
            pps_b64 = base64.b64encode(self.pps).decode("ascii")
            return f"{sps_b64},{pps_b64}"
        # Standard baseline 480p fallback parameters
        return "Z00AKp2oHgCJ+WbgICAgQA==,aO48gA=="


class RTPPacketizer:
    """RFC 6184 RTP Packetizer for H.264 NAL units."""

    def __init__(self, ssrc: int = 0x12345678, mtu: int = 1400):
        self.ssrc = ssrc
        self.mtu = mtu
        self.seq_num = 0

    def packetize(
        self, nalus: List[bytes], timestamp: int
    ) -> List[bytes]:
        packets = []
        for idx, nalu in enumerate(nalus):
            is_last_nalu = (idx == len(nalus) - 1)

            if len(nalu) <= self.mtu:
                # Single NAL unit packet
                m = 1 if is_last_nalu else 0
                header = struct.pack(
                    "!BBHII",
                    0x80,
                    (m << 7) | 96,
                    self.seq_num & 0xFFFF,
                    timestamp & 0xFFFFFFFF,
                    self.ssrc,
                )
                packets.append(header + nalu)
                self.seq_num = (self.seq_num + 1) & 0xFFFF
            else:
                # FU-A Fragmentation
                nal_header = nalu[0]
                fu_indicator = (nal_header & 0xE0) | 28
                nal_type = nal_header & 0x1F
                payload = nalu[1:]
                max_chunk = self.mtu - 2
                offset = 0
                total_len = len(payload)

                while offset < total_len:
                    end = min(offset + max_chunk, total_len)
                    is_start = 1 if offset == 0 else 0
                    is_end = 1 if end == total_len else 0
                    m = 1 if (is_last_nalu and is_end) else 0

                    fu_header = (is_start << 7) | (is_end << 6) | nal_type
                    rtp_header = struct.pack(
                        "!BBHII",
                        0x80,
                        (m << 7) | 96,
                        self.seq_num & 0xFFFF,
                        timestamp & 0xFFFFFFFF,
                        self.ssrc,
                    )
                    packets.append(
                        rtp_header + bytes([fu_indicator, fu_header]) + payload[offset:end]
                    )
                    self.seq_num = (self.seq_num + 1) & 0xFFFF
                    offset = end

        return packets


class RTSPClientSession:
    """Manages state for an individual connected RTSP client."""

    def __init__(self, session_id: str, client_ip: str, writer: asyncio.StreamWriter):
        self.session_id = session_id
        self.client_ip = client_ip
        self.writer = writer
        self.transport_mode: str = "udp"  # "udp" or "tcp"
        self.client_rtp_port: int = 0
        self.client_rtcp_port: int = 0
        self.server_rtp_sock: Optional[socket.socket] = None
        self.server_rtcp_sock: Optional[socket.socket] = None
        self.server_rtp_port: int = 0
        self.server_rtcp_port: int = 0
        self.interleaved_rtp: int = 0
        self.interleaved_rtcp: int = 1
        self.is_playing: bool = False

    def close(self):
        self.is_playing = False
        if self.server_rtp_sock:
            try:
                self.server_rtp_sock.close()
            except Exception:
                pass
            self.server_rtp_sock = None
        if self.server_rtcp_sock:
            try:
                self.server_rtcp_sock.close()
            except Exception:
                pass
            self.server_rtcp_sock = None


class RTSPServer:
    """Async RTSP 1.0 Server (RFC 2326) handling OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN."""

    def __init__(self, host: str, port: int, camera_hub: CameraHub, encoder: H264Encoder):
        self.host = host
        self.port = port
        self.camera_hub = camera_hub
        self.encoder = encoder
        self.packetizer = RTPPacketizer()
        self.sessions: Dict[str, RTSPClientSession] = {}
        self.running = threading.Event()
        self.server: Optional[asyncio.Server] = None
        self.next_session_id = 1000
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _allocate_udp_ports(self) -> Tuple[socket.socket, socket.socket, int, int]:
        """Binds an even/odd port pair for RTP/RTCP."""
        for p in range(20000, 30000, 2):
            try:
                s_rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s_rtp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s_rtp.bind(("0.0.0.0", p))

                s_rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s_rtcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s_rtcp.bind(("0.0.0.0", p + 1))
                return s_rtp, s_rtcp, p, p + 1
            except OSError:
                continue
        raise RuntimeError("No available UDP port pair for RTSP/RTP.")

    def _generate_sdp(self) -> str:
        sprop = self.encoder.get_sprop_parameter_sets()
        w = self.camera_hub.width
        h = self.camera_hub.height
        sdp = (
            "v=0\r\n"
            f"o=- 0 0 IN IP4 {self.host}\r\n"
            "s=Raspberry Pi Camera RTSP Stream\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "t=0 0\r\n"
            "m=video 0 RTP/AVP 96\r\n"
            "a=rtpmap:96 H264/90000\r\n"
            f"a=fmtp:96 packetization-mode=1;sprop-parameter-sets={sprop}\r\n"
            f"a=framesize:96 {w}-{h}\r\n"
            "a=control:track1\r\n"
        )
        return sdp

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        client_ip = peer[0] if peer else "unknown"
        session: Optional[RTSPClientSession] = None

        try:
            while self.running.is_set():
                first_byte = await reader.read(1)
                if not first_byte:
                    break

                if first_byte == b"$":
                    # Interleaved binary data from client (e.g. RTCP Receiver Report)
                    interleaved_hdr = await reader.readexactly(3)
                    pkt_len = struct.unpack("!H", interleaved_hdr[1:3])[0]
                    if pkt_len > 0:
                        await reader.readexactly(pkt_len)
                    continue

                rest = await reader.readuntil(b"\r\n\r\n")
                data = first_byte + rest
                request_text = data.decode("utf-8", errors="ignore")
                lines = [l.strip() for l in request_text.split("\r\n") if l.strip()]
                if not lines:
                    continue

                request_line = lines[0]
                headers = {}
                for line in lines[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip().lower()] = v.strip()

                parts = request_line.split()
                if len(parts) < 2:
                    continue
                method, url = parts[0], parts[1]
                cseq = headers.get("cseq", "1")
                print(f"[RTSP] {client_ip} -> {method} {url} (CSeq: {cseq})")

                if method == "OPTIONS":
                    resp = (
                        "RTSP/1.0 200 OK\r\n"
                        f"CSeq: {cseq}\r\n"
                        "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, PAUSE\r\n"
                        "\r\n"
                    )
                    writer.write(resp.encode())
                    await writer.drain()

                elif method == "DESCRIBE":
                    sdp = self._generate_sdp()
                    resp = (
                        "RTSP/1.0 200 OK\r\n"
                        f"CSeq: {cseq}\r\n"
                        f"Content-Base: rtsp://{self.host}:{self.port}/live/\r\n"
                        "Content-Type: application/sdp\r\n"
                        f"Content-Length: {len(sdp.encode())}\r\n"
                        "\r\n"
                        f"{sdp}"
                    )
                    writer.write(resp.encode())
                    await writer.drain()

                elif method == "SETUP":
                    if session is None:
                        self.next_session_id += 1
                        sess_id = str(self.next_session_id)
                        session = RTSPClientSession(sess_id, client_ip, writer)
                        self.sessions[sess_id] = session

                    transport_hdr = headers.get("transport", "")
                    if "RTP/AVP/TCP" in transport_hdr or "interleaved" in transport_hdr:
                        # TCP Interleaved mode
                        session.transport_mode = "tcp"
                        m = re.search(r"interleaved=(\d+)-(\d+)", transport_hdr)
                        if m:
                            session.interleaved_rtp = int(m.group(1))
                            session.interleaved_rtcp = int(m.group(2))
                        else:
                            session.interleaved_rtp = 0
                            session.interleaved_rtcp = 1

                        resp_trans = f"RTP/AVP/TCP;unicast;interleaved={session.interleaved_rtp}-{session.interleaved_rtcp}"
                    else:
                        # UDP Unicast mode
                        session.transport_mode = "udp"
                        m = re.search(r"client_port=(\d+)-(\d+)", transport_hdr)
                        if m:
                            session.client_rtp_port = int(m.group(1))
                            session.client_rtcp_port = int(m.group(2))
                        else:
                            session.client_rtp_port = 5004
                            session.client_rtcp_port = 5005

                        s_rtp, s_rtcp, p_rtp, p_rtcp = self._allocate_udp_ports()
                        session.server_rtp_sock = s_rtp
                        session.server_rtcp_sock = s_rtcp
                        session.server_rtp_port = p_rtp
                        session.server_rtcp_port = p_rtcp

                        resp_trans = (
                            f"RTP/AVP;unicast;client_port={session.client_rtp_port}-{session.client_rtcp_port};"
                            f"server_port={session.server_rtp_port}-{session.server_rtcp_port}"
                        )

                    resp = (
                        "RTSP/1.0 200 OK\r\n"
                        f"CSeq: {cseq}\r\n"
                        f"Session: {session.session_id};timeout=60\r\n"
                        f"Transport: {resp_trans}\r\n"
                        "\r\n"
                    )
                    writer.write(resp.encode())
                    await writer.drain()

                elif method == "PLAY":
                    if session:
                        session.is_playing = True
                        self.camera_hub.active_clients_count += 1
                        print(f"[RTSP] Client {client_ip} started playback via {session.transport_mode.upper()}")
                        resp = (
                            "RTSP/1.0 200 OK\r\n"
                            f"CSeq: {cseq}\r\n"
                            f"Session: {session.session_id}\r\n"
                            "Range: npt=0.000-\r\n"
                            f"RTP-Info: url=rtsp://{self.host}:{self.port}/live/track1;seq=0;rtptime=0\r\n"
                            "\r\n"
                        )
                        writer.write(resp.encode())
                        await writer.drain()

                elif method == "TEARDOWN":
                    if session:
                        session.is_playing = False
                        self.camera_hub.active_clients_count = max(
                            0, self.camera_hub.active_clients_count - 1
                        )
                    resp = (
                        "RTSP/1.0 200 OK\r\n"
                        f"CSeq: {cseq}\r\n"
                        "\r\n"
                    )
                    writer.write(resp.encode())
                    await writer.drain()
                    break

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            print(f"[RTSP] Client error: {e}")
        finally:
            if session:
                if session.is_playing:
                    self.camera_hub.active_clients_count = max(
                        0, self.camera_hub.active_clients_count - 1
                    )
                session.close()
                self.sessions.pop(session.session_id, None)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            print(f"[RTSP] Client disconnected: {client_ip}")

    def _broadcast_rtp_tcp(self, packets: List[bytes]):
        for session in list(self.sessions.values()):
            if not session.is_playing:
                continue
            if session.transport_mode == "tcp" and session.writer:
                if session.writer.is_closing():
                    session.is_playing = False
                    continue
                for pkt in packets:
                    if not session.is_playing:
                        break
                    try:
                        hdr = struct.pack("!BBH", 0x24, session.interleaved_rtp, len(pkt))
                        session.writer.write(hdr + pkt)
                    except Exception:
                        session.is_playing = False
                        break

    def broadcast_rtp_packets(self, packets: List[bytes]):
        """Transmits RTP packets to all playing RTSP clients via UDP or TCP."""
        has_tcp = False
        for session in list(self.sessions.values()):
            if not session.is_playing:
                continue

            if session.transport_mode == "udp" and session.server_rtp_sock:
                for pkt in packets:
                    try:
                        session.server_rtp_sock.sendto(
                            pkt, (session.client_ip, session.client_rtp_port)
                        )
                    except Exception:
                        pass
            elif session.transport_mode == "tcp":
                has_tcp = True

        if has_tcp and self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._broadcast_rtp_tcp, packets)


class RTSPStreamingServer:
    """Master server orchestrating CameraHub, H264Encoder, RTSP server, and HTTP Web Viewer."""

    def __init__(
        self,
        host: str,
        rtsp_port: int,
        http_port: int,
        fps: int,
        resolution: int,
        quality: int = 60,
    ):
        self.host = host
        self.rtsp_port = rtsp_port
        self.http_port = http_port
        self.fps = fps
        self.resolution = resolution
        self.quality = quality
        self.width, self.height = RESOLUTION_MAP[resolution]

        self.camera_hub = CameraHub(fps=fps, resolution=resolution)
        self.encoder = H264Encoder(width=self.width, height=self.height, fps=fps)
        self.rtsp_server = RTSPServer(
            host=host, port=rtsp_port, camera_hub=self.camera_hub, encoder=self.encoder
        )

        self.app = web.Application()
        self.web_clients = set()
        self.running = threading.Event()
        self.streaming_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _broadcast_to_web_queues(self, jpeg_bytes: bytes):
        for q in list(self.web_clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(jpeg_bytes)
            except asyncio.QueueFull:
                pass

    def _streaming_worker(self):
        """Worker loop: grabs frames from CameraHub, encodes H.264, and feeds RTSP & WebSocket."""
        pts = 0
        pts_step = 90000 // self.fps
        target_interval = 1.0 / self.fps
        last_time = time.perf_counter()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]

        while self.running.is_set():
            loop_start = time.perf_counter()
            frame = self.camera_hub.get_latest_frame()

            if frame is not None:
                # 1. H.264 Encode & RTP Broadcast if active RTSP clients exist
                if len(self.rtsp_server.sessions) > 0:
                    nalus = self.encoder.encode(frame, pts)
                    if nalus:
                        packets = self.rtsp_server.packetizer.packetize(nalus, pts)
                        self.rtsp_server.broadcast_rtp_packets(packets)
                    pts = (pts + pts_step) & 0xFFFFFFFF

                # 2. Fast JPEG compress & dispatch to WebSocket clients
                if len(self.web_clients) > 0:
                    success, buffer = cv2.imencode(".jpg", frame, encode_params)
                    if success:
                        jpeg_bytes = buffer.tobytes()
                        if self.loop and not self.loop.is_closed():
                            self.loop.call_soon_threadsafe(
                                self._broadcast_to_web_queues, jpeg_bytes
                            )

            # Pacing
            work_time = time.perf_counter() - loop_start
            sleep_duration = target_interval - work_time
            if sleep_duration > 0.001:
                time.sleep(sleep_duration)

    async def index(self, request):
        html_path = os.path.join(os.path.dirname(__file__), "web", "rstp.html")
        if not os.path.exists(html_path):
            return web.Response(
                text="<h1>web/rstp.html not found</h1>", content_type="text/html"
            )
        return web.FileResponse(html_path)

    async def ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        q = asyncio.Queue(maxsize=1)
        self.web_clients.add(q)
        self.camera_hub.active_clients_count += 1
        print(f"[Web Viewer] Client connected: {request.remote}")

        async def send_loop():
            while not ws.closed:
                try:
                    data = await q.get()
                    if ws.closed:
                        break
                    await ws.send_bytes(data)
                except (asyncio.CancelledError, ConnectionResetError):
                    break

        send_task = asyncio.create_task(send_loop())
        try:
            async for _ in ws:
                pass
        except Exception:
            pass
        finally:
            send_task.cancel()
            self.web_clients.discard(q)
            self.camera_hub.active_clients_count = max(0, self.camera_hub.active_clients_count - 1)
            print(f"[Web Viewer] Client disconnected: {request.remote}")
        return ws

    async def stats_handler(self, request):
        data = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "rtsp_clients": len(self.rtsp_server.sessions),
            "web_clients": len(self.web_clients),
            "rtsp_url": f"rtsp://{self.host}:{self.rtsp_port}/live",
        }
        return web.json_response(data, headers={"Access-Control-Allow-Origin": "*"})

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.rtsp_server.loop = self.loop
        self.running.set()
        self.rtsp_server.running.set()
        self.camera_hub.start()

        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

        # Start RTSP async server
        self.rtsp_server.server = await asyncio.start_server(
            self.rtsp_server.handle_client, self.host, self.rtsp_port
        )
        print(f"[RTSP] Server running on rtsp://{self.host}:{self.rtsp_port}/live")

        # Configure HTTP app routes
        self.app.router.add_get("/", self.index)
        self.app.router.add_get("/rstp.html", self.index)
        self.app.router.add_get("/rtsp.html", self.index)
        self.app.router.add_get("/ws", self.ws_handler)
        self.app.router.add_get("/stats", self.stats_handler)

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.http_port)
        await site.start()
        print(f"[HTTP] Web viewer running at http://{self.host}:{self.http_port}/")

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            self.running.clear()
            self.rtsp_server.running.clear()
            if self.rtsp_server.server:
                self.rtsp_server.server.close()
                await self.rtsp_server.server.wait_closed()
            await runner.cleanup()
            self.camera_hub.stop()
            print("Server stopped cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi RTSP Video Broadcasting Server & Web Viewer"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host IP address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--rtsp-port",
        type=int,
        default=8554,
        help="RTSP server port (default: 8554)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP web viewer port (default: 8080)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        choices=[30, 90],
        default=30,
        help="Target framerate: 30 or 90 FPS (default: 30)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[480, 720, 1080],
        default=480,
        help="Vertical resolution: 480 (640x480), 720 (1280x720), or 1080 (1920x1080) (default: 480)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=60,
        help="JPEG quality for web preview (1-100, default: 60)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server = RTSPStreamingServer(
        host=args.host,
        rtsp_port=args.rtsp_port,
        http_port=args.http_port,
        fps=args.fps,
        resolution=args.resolution,
        quality=args.quality,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user.")


if __name__ == "__main__":
    main()
