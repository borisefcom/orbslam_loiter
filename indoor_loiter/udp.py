import json, socket, errno, time

def parse_hostport(s: str):
    host, port = s.rsplit(":", 1)
    return host, int(port)

class UdpJsonTx:
    def __init__(self, host: str, port: int):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Telemetry/overlay is best-effort; never allow a congested network to stall
        # tracker/control threads. Drop packets if kernel buffers are full.
        try:
            self.sock.setblocking(False)
        except Exception:
            pass

    def send(self, payload: dict):
        try:
            buf = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        except Exception:
            return
        try:
            self.sock.sendto(buf, self.addr)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as e:
            if getattr(e, "errno", None) in (errno.EWOULDBLOCK, errno.EAGAIN, errno.ENOBUFS):
                return
            return

class UdpJsonRx:
    def __init__(self, host: str, port: int, bufsize: int = 65535):
        self.addr = (host, int(port))
        self.bufsize = bufsize
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Reuse flags before bind to avoid EADDRINUSE after quick restarts
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except Exception:
            pass
        try:
            self.sock.bind(self.addr)
        except OSError as e:
            # If another run is in teardown, retry once
            if e.errno == errno.EADDRINUSE:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except Exception:
                    pass
                try:
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except Exception:
                    pass
                self.sock.bind(self.addr)
            else:
                raise

    def __iter__(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(self.bufsize)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                # Keep the iterator alive even if the socket hiccups.
                try:
                    time.sleep(0.01)
                except Exception:
                    pass
                continue
            try:
                yield json.loads(data.decode("utf-8", errors="ignore")), addr
            except Exception:
                continue
