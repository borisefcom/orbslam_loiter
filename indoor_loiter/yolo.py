import time
import threading
from typing import List
import numpy as np
import cv2

from tracker_types import Det  # local types

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAIL = True
except Exception as e:
    print("[WARN] Ultralytics not available:", e)
    YOLO_AVAIL = False

class YoloThread(threading.Thread):
    def __init__(self, enabled: bool, model_path: str, det_period: float, conf: float, iou: float,
                 max_det: int, device: str, scale: float):
        super().__init__(daemon=True)
        self.enabled = bool(enabled) and YOLO_AVAIL
        self.model_path = model_path
        self.model = _YOLO(model_path) if self.enabled else None
        self.det_period = float(det_period)
        self.conf = float(conf)
        self.iou  = float(iou)
        self.max_det = int(max_det)
        self.device = device
        self.scale = float(scale)
        self.running = True
        self._frame = None
        self._size = (0,0)
        self._dets: List[Det] = []
        self.lock = threading.Lock()
        self._last_run_dt = 0.0
        self._evt = threading.Event()

    def stop(self, timeout: float = 1.0):
        # Signal thread to exit and optionally wait a bit.
        self.running = False
        try:
            self._evt.set()
        except Exception:
            pass
        try:
            self.join(timeout=timeout)
        except RuntimeError:
            # Thread was never started
            pass

    def set_enabled(self, on: bool) -> bool:
        """Toggle inference; lazily loads model when enabling."""
        if not YOLO_AVAIL:
            self.enabled = False
            return False
        if on and self.model is None:
            try:
                self.model = _YOLO(self.model_path)
            except Exception as e:
                print("[YOLO] enable failed:", e)
                self.enabled = False
                return False
        self.enabled = bool(on) and (self.model is not None)
        if not self.enabled:
            with self.lock:
                self._dets = []
                self._frame = None
        return self.enabled

    def stats(self):
        with self.lock:
            return {"last_run_s": float(self._last_run_dt), "det_period_s": float(self.det_period)}

    def update(self, frame_bgr, W, H):
        if not self.enabled or self.model is None:
            return
        if not self.lock.acquire(blocking=False):
            return  # skip if worker is copying; never block tracker loop
        try:
            self._frame = frame_bgr
            self._size = (W,H)
            self._evt.set()
        finally:
            self.lock.release()

    def get(self) -> List[Det]:
        if not self.lock.acquire(blocking=False):
            return []  # avoid blocking caller; return empty if busy
        try:
            return list(self._dets)
        finally:
            self.lock.release()

    def run(self):
        last_t = 0.0
        while self.running:
            if not self.enabled or self.model is None:
                self._evt.wait(timeout=0.05)
                continue
            # wait for either det_period or new frame event
            self._evt.wait(timeout=self.det_period)
            self._evt.clear()
            if not self.running:
                break
            now = time.time()
            if (now - last_t) < self.det_period:
                continue
            last_t = now
            with self.lock:
                # Use shared frame reference (read-only) to avoid extra copies; FrameBus provides BGR.
                f = self._frame
                W,H = self._size
            if f is None:
                continue
            infer = f
            s = self.scale
            if 0.0 < s < 1.0:
                infer = cv2.resize(f, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
            try:
                t0 = time.time()
                res = self.model.predict(source=infer, imgsz=640, conf=self.conf, iou=self.iou,
                                         device=self.device, max_det=self.max_det, verbose=False)
                dets=[]
                if res and hasattr(res[0],"boxes") and res[0].boxes is not None:
                    r0 = res[0]; boxes=r0.boxes
                    xyxy=boxes.xyxy; conf=boxes.conf; cls=boxes.cls
                    if hasattr(xyxy,"cpu"): xyxy=xyxy.cpu()
                    if hasattr(conf,"cpu"): conf=conf.cpu()
                    if hasattr(cls,"cpu"): cls=cls.cpu()
                    xyxy=xyxy.numpy(); conf=conf.numpy(); cls=cls.numpy().astype(int)
                    names=getattr(r0,"names",{}) or {}
                    for i in range(min(len(xyxy), self.max_det)):
                        x1,y1,x2,y2=[float(v) for v in xyxy[i]]
                        if 0.0 < s < 1.0:
                            x1/=s; y1/=s; x2/=s; y2/=s
                        w=max(0.0,x2-x1); h=max(0.0,y2-y1)
                        dets.append(Det(
                            bbox=(int(round(x1)),int(round(y1)),int(round(w)),int(round(h))),
                            conf=float(conf[i]), cls=int(cls[i]),
                            label=str(names.get(int(cls[i]), int(cls[i])))))
                with self.lock:
                    self._dets = dets
                    self._last_run_dt = time.time() - t0
            except Exception as e:
                print("[DETECT] error:", e)
