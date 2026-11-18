import os
import json
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


class TableMonitorEngine:
    def __init__(self, zones_path: str, model_path: str = "yolov8n.pt", fps: float = 30.0):
        with open(zones_path, "r") as f:
            self.raw_zones = json.load(f)

        self.fps = fps if fps > 0 else 30.0
        self.model = YOLO(model_path)

        # Inisialisasi PolygonZone dari Supervision untuk setiap meja
        self.zones = {}
        for table_id, coords in self.raw_zones.items():
            polygon = np.array(coords, dtype=np.int32)
            self.zones[table_id] = sv.PolygonZone(polygon=polygon)

        # State machine & metrik per meja
        self.table_states = {}
        for table_id in self.raw_zones.keys():
            self.table_states[table_id] = {
                "table_id": table_id,
                "status": "FREE",               # FREE | OCCUPIED | NEEDS_CLEANING
                "guest_count": 0,
                "occupied_seconds": 0.0,
                "presence_seconds": 0.0,        # Akumulasi waktu ada orang sebelum sah OCCUPIED
                "cleaning_seconds": 0.0,        # Akumulasi waktu staf membersihkan meja
                "initial_tracker_ids": set(),   # ID tamu awal untuk mendeteksi kedatangan pelayan
                "waiter_attended": False,
                "alerts": [],
            }

        # Parameter Threshold (disesuaikan agar responsif untuk demo video 10 detik)
        self.SIT_CONFIRM_SEC = 1.2       # Waktu minimum orang di zona agar sah dianggap duduk (bukan lewat)
        self.UNATTENDED_ALERT_SEC = 4.0  # Batas waktu tamu menunggu pelayan (simulasi 5 menit)
        self.CLEAN_CONFIRM_SEC = 2.0     # Waktu staf di zona meja kotor untuk reset ke FREE

        # Warna BGR untuk OpenCV
        self.STATUS_COLORS = {
            "FREE": (46, 204, 113),            # Hijau
            "OCCUPIED": (60, 76, 231),         # Merah
            "NEEDS_CLEANING": (15, 196, 241),  # Kuning/Oranye
        }

    def process_frame(self, frame: np.ndarray):
        dt = 1.0 / self.fps

        # 1. Jalankan YOLOv8 + ByteTrack (hanya deteksi 'person' -> class 0)
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.35,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # 2. Evaluasi setiap zona meja
        active_alerts = []
        for table_id, zone in self.zones.items():
            state = self.table_states[table_id]
            state["alerts"] = []

            # Cek deteksi mana saja yang masuk ke dalam poligon meja ini
            mask = zone.trigger(detections=detections)
            in_zone_detections = detections[mask]

            person_count = len(in_zone_detections)
            current_ids = set()
            if in_zone_detections.tracker_id is not None:
                current_ids = set(int(tid) for tid in in_zone_detections.tracker_id)

            # --- LOGIKA STATE MACHINE ---
            if state["status"] == "FREE":
                if person_count > 0:
                    state["presence_seconds"] += dt
                    if state["presence_seconds"] >= self.SIT_CONFIRM_SEC:
                        state["status"] = "OCCUPIED"
                        state["guest_count"] = person_count
                        state["occupied_seconds"] = state["presence_seconds"]
                        state["initial_tracker_ids"] = current_ids.copy()
                        state["waiter_attended"] = False
                else:
                    # Reset jika orang hanya lewat sekilas
                    state["presence_seconds"] = 0.0
                    state["guest_count"] = 0

            elif state["status"] == "OCCUPIED":
                if person_count > 0:
                    state["occupied_seconds"] += dt
                    state["guest_count"] = person_count

                    # Cek apakah ada ID baru masuk (indikasi pelayan datang menghampiri meja)
                    new_ids = current_ids - state["initial_tracker_ids"]
                    if len(new_ids) > 0:
                        state["waiter_attended"] = True

                    # Trigger alert jika tamu sudah duduk lama tapi belum dihampiri pelayan
                    if not state["waiter_attended"] and state["occupied_seconds"] >= self.UNATTENDED_ALERT_SEC:
                        msg = f"{table_id}: Tamu menunggu pelayan ({int(state['occupied_seconds'])}s)"
                        state["alerts"].append(msg)
                        active_alerts.append({"table_id": table_id, "type": "WAITER", "message": msg})
                else:
                    # Semua tamu meninggalkan meja -> berubah ke NEEDS_CLEANING
                    state["status"] = "NEEDS_CLEANING"
                    state["guest_count"] = 0
                    state["cleaning_seconds"] = 0.0

            elif state["status"] == "NEEDS_CLEANING":
                msg = f"{table_id}: Meja kosong, segera bersihkan!"
                state["alerts"].append(msg)
                active_alerts.append({"table_id": table_id, "type": "CLEANING", "message": msg})

                # Jika staf mendekat & membersihkan meja selama CLEAN_CONFIRM_SEC -> reset ke FREE
                if person_count > 0:
                    state["cleaning_seconds"] += dt
                    if state["cleaning_seconds"] >= self.CLEAN_CONFIRM_SEC:
                        state["status"] = "FREE"
                        state["occupied_seconds"] = 0.0
                        state["presence_seconds"] = 0.0
                        state["cleaning_seconds"] = 0.0
                        state["initial_tracker_ids"] = set()
                        state["waiter_attended"] = False
                else:
                    state["cleaning_seconds"] = 0.0

        # 3. Gambar anotasi visual pada frame
        annotated_frame = self._draw_overlays(frame.copy(), detections)

        # 4. Hitung ringkasan & prediksi turnover antrean
        summary = self._build_summary(active_alerts)
        return annotated_frame, self.table_states, summary

    def _draw_overlays(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        overlay = frame.copy()

        # Gambar poligon transparan + garis tepi per meja
        for table_id, coords in self.raw_zones.items():
            state = self.table_states[table_id]
            status = state["status"]
            color = self.STATUS_COLORS[status]
            pts = np.array(coords, dtype=np.int32)

            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # Label status di sudut atas poligon
            x_min, y_min = pts.min(axis=0)
            label = f"{table_id} | {status}"
            if status == "OCCUPIED":
                label += f" ({state['guest_count']}p - {int(state['occupied_seconds'])}s)"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x_min, max(0, y_min - th - 8)), (x_min + tw + 8, y_min), color, -1)
            cv2.putText(
                frame,
                label,
                (x_min + 4, max(12, y_min - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Blend transparansi poligon meja
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Gambar titik pelacakan orang (foot anchor) & ID
        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cx, cy = int((x1 + x2) / 2), int(y2)
            tid = detections.tracker_id[i] if detections.tracker_id is not None else -1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            if tid != -1:
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 0),
                    1,
                )

        return frame

    def _build_summary(self, active_alerts: list) -> dict:
        total = len(self.table_states)
        free_count = sum(1 for s in self.table_states.values() if s["status"] == "FREE")
        occupied_count = sum(1 for s in self.table_states.values() if s["status"] == "OCCUPIED")
        cleaning_count = sum(1 for s in self.table_states.values() if s["status"] == "NEEDS_CLEANING")
        total_guests = sum(s["guest_count"] for s in self.table_states.values())

        # Urutkan meja OCCUPIED berdasarkan durasi duduk terlama (prediksi paling cepat kosong)
        occupied_tables = [
            s for s in self.table_states.values() if s["status"] == "OCCUPIED"
        ]
        occupied_tables.sort(key=lambda x: x["occupied_seconds"], reverse=True)
        next_available = occupied_tables[0]["table_id"] if occupied_tables else "-"

        return {
            "total_tables": total,
            "free": free_count,
            "occupied": occupied_count,
            "needs_cleaning": cleaning_count,
            "total_guests": total_guests,
            "occupancy_rate": round((occupied_count / total) * 100, 1) if total > 0 else 0.0,
            "next_available_table": next_available,
            "alerts": active_alerts,
        }