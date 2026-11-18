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

        # Perluas area poligon meja (1.35x) agar mencakup kursi di sekelilingnya
        self.expanded_zones = {}
        for table_id, coords in self.raw_zones.items():
            self.expanded_zones[table_id] = self._expand_polygon(
                np.array(coords, dtype=np.float32), scale=1.35
            )

        # State machine & metrik per meja
        self.table_states = {}
        for table_id in self.raw_zones.keys():
            self.table_states[table_id] = {
                "table_id": table_id,
                "status": "FREE",               # FREE | OCCUPIED | NEEDS_CLEANING
                "guest_count": 0,
                "occupied_seconds": 0.0,
                "presence_seconds": 0.0,
                "empty_seconds": 0.0,
                "cleaning_seconds": 0.0,
                "initial_tracker_ids": set(),
                "waiter_attended": False,
                "alerts": [],
            }

        # Parameter Threshold
        self.SIT_CONFIRM_SEC = 0.4       # Waktu minimum orang di zona agar sah dianggap duduk
        self.EMPTY_CONFIRM_SEC = 1.0     # Jeda toleransi kosong sebelum sah dianggap NEEDS_CLEANING
        self.UNATTENDED_ALERT_SEC = 4.0  # Batas waktu tamu menunggu pelayan
        self.CLEAN_CONFIRM_SEC = 2.0     # Waktu staf di zona meja kotor untuk reset ke FREE

        # Warna BGR untuk OpenCV
        self.STATUS_COLORS = {
            "FREE": (46, 204, 113),            # Hijau
            "OCCUPIED": (60, 76, 231),         # Merah
            "NEEDS_CLEANING": (15, 196, 241),  # Kuning/Oranye
        }

    def _expand_polygon(self, pts: np.ndarray, scale: float = 1.35) -> np.ndarray:
        """Memperluas poligon dari titik pusatnya agar mencakup kursi di sekitar meja."""
        centroid = pts.mean(axis=0)
        expanded = centroid + (pts - centroid) * scale
        return expanded.astype(np.int32)

    def _get_assigned_table(self, bbox: np.ndarray):
        """Menentukan meja mana yang ditempati seseorang (1 orang maksimal hanya 1 meja terdekat)."""
        x1, y1, x2, y2 = bbox
        check_points = [
            (float((x1 + x2) / 2), float((y1 + y2) / 2)),
            (float((x1 + x2) / 2), float(y1 + (y2 - y1) * 0.4)),
        ]
        person_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        matched_tables = []
        for table_id, poly_pts in self.expanded_zones.items():
            for pt in check_points:
                if cv2.pointPolygonTest(poly_pts, pt, False) >= 0:
                    table_center = poly_pts.mean(axis=0)
                    dist = np.linalg.norm(person_center - table_center)
                    matched_tables.append((dist, table_id))
                    break

        if not matched_tables:
            return None

        matched_tables.sort(key=lambda x: x[0])
        return matched_tables[0][1]

    def process_frame(self, frame: np.ndarray):
        dt = 1.0 / self.fps

        # 1. Deteksi & pelacakan objek 'person' (class 0)
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.30,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0:
            widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            valid_person_mask = (widths >= 35) & (heights >= 60)
            detections = detections[valid_person_mask]

        # 3. Petakan setiap orang yang valid ke maksimal 1 meja terdekat
        table_detections = {tid: [] for tid in self.expanded_zones.keys()}
        for i in range(len(detections)):
            assigned_tid = self._get_assigned_table(detections.xyxy[i])
            if assigned_tid is not None:
                table_detections[assigned_tid].append(i)

        # 4. Evaluasi State Machine per meja
        active_alerts = []
        for table_id in self.expanded_zones.keys():
            state = self.table_states[table_id]
            state["alerts"] = []

            in_table_indices = table_detections[table_id]
            person_count = len(in_table_indices)
            current_ids = set()
            if detections.tracker_id is not None:
                for idx in in_table_indices:
                    current_ids.add(int(detections.tracker_id[idx]))

            if state["status"] == "FREE":
                if person_count > 0:
                    state["presence_seconds"] += dt
                    if state["presence_seconds"] >= self.SIT_CONFIRM_SEC:
                        state["status"] = "OCCUPIED"
                        state["guest_count"] = person_count
                        state["occupied_seconds"] = state["presence_seconds"]
                        state["empty_seconds"] = 0.0
                        state["initial_tracker_ids"] = current_ids.copy()
                        state["waiter_attended"] = False
                else:
                    state["presence_seconds"] = 0.0
                    state["guest_count"] = 0

            elif state["status"] == "OCCUPIED":
                if person_count > 0:
                    state["empty_seconds"] = 0.0
                    state["occupied_seconds"] += dt
                    state["guest_count"] = person_count

                    new_ids = current_ids - state["initial_tracker_ids"]
                    if len(new_ids) > 0:
                        state["waiter_attended"] = True

                    if not state["waiter_attended"] and state["occupied_seconds"] >= self.UNATTENDED_ALERT_SEC:
                        msg = f"{table_id}: Tamu menunggu pelayan ({int(state['occupied_seconds'])}s)"
                        state["alerts"].append(msg)
                        active_alerts.append({"table_id": table_id, "type": "WAITER", "message": msg})
                else:
                    state["empty_seconds"] += dt
                    if state["empty_seconds"] >= self.EMPTY_CONFIRM_SEC:
                        state["status"] = "NEEDS_CLEANING"
                        state["guest_count"] = 0
                        state["cleaning_seconds"] = 0.0

            elif state["status"] == "NEEDS_CLEANING":
                msg = f"{table_id}: Meja kosong, segera bersihkan!"
                state["alerts"].append(msg)
                active_alerts.append({"table_id": table_id, "type": "CLEANING", "message": msg})

                if person_count > 0:
                    state["cleaning_seconds"] += dt
                    if state["cleaning_seconds"] >= self.CLEAN_CONFIRM_SEC:
                        state["status"] = "FREE"
                        state["occupied_seconds"] = 0.0
                        state["presence_seconds"] = 0.0
                        state["empty_seconds"] = 0.0
                        state["cleaning_seconds"] = 0.0
                        state["initial_tracker_ids"] = set()
                        state["waiter_attended"] = False
                else:
                    state["cleaning_seconds"] = 0.0

        # 5. Gambar overlay visual & susun ringkasan metrik
        annotated_frame = self._draw_overlays(frame.copy(), detections)
        summary = self._build_summary(active_alerts)
        return annotated_frame, self.table_states, summary

    def _draw_overlays(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        overlay = frame.copy()

        for table_id, coords in self.raw_zones.items():
            state = self.table_states[table_id]
            status = state["status"]
            color = self.STATUS_COLORS[status]
            pts = np.array(coords, dtype=np.int32)

            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # Letakkan label status di tengah meja agar rapi dan tidak bertabrakan dengan kursi
            cx_lbl, cy_lbl = pts.mean(axis=0).astype(int)
            label = f"{table_id} | {status}"
            if status == "OCCUPIED":
                label += f" ({state['guest_count']}p - {int(state['occupied_seconds'])}s)"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lbl_x = max(0, cx_lbl - tw // 2)
            lbl_y = max(th + 6, cy_lbl)

            cv2.rectangle(frame, (lbl_x - 3, lbl_y - th - 4), (lbl_x + tw + 3, lbl_y + 3), color, -1)
            cv2.putText(
                frame,
                label,
                (lbl_x, lbl_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            tid = detections.tracker_id[i] if detections.tracker_id is not None else -1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            if tid != -1:
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
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