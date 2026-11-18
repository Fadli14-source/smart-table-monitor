# 🍽️ AI Smart Table Occupancy & Operations Monitor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![Supervision](https://img.shields.io/badge/Supervision-PolygonZone-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)

Real-time Computer Vision floor monitoring, queue turnover prediction, and operational alerting system designed for **Food & Beverage (F&B)** businesses using existing CCTV infrastructure.

---

## 🎯 Business Problem & Target Users

High-traffic restaurants, cafes, and food courts frequently lose revenue and customer satisfaction due to manual floor monitoring:
* **Slow Table Turnover:** Bussing staff miss newly vacated tables, leaving dirty dishes uncleared while waiting guests stand in line.
* **Delayed Guest Service:** Newly seated customers wait too long during peak hours before a waiter approaches with a menu.
* **Unpredictable Queue Estimates:** Front-of-house greeters lack real-time visibility into which occupied table has been seated longest.
* **High IoT Hardware Costs:** Traditional smart-table systems require expensive under-table sensors. This solution leverages **existing overhead CCTV cameras**.

**Target Users:** Restaurant Operations Managers, Front-of-House Staff (Greeters, Waiters, Bussers), and F&B Business Owners.

---

## ✨ Core Features

1. **Real-Time 3-State Table Monitoring:**
   * 🟢 **`FREE`:** Clean and ready for incoming guests.
   * 🔴 **`OCCUPIED`:** Active guests seated (tracks guest count and dwell duration).
   * 🟡 **`NEEDS_CLEANING`:** Triggered automatically once guests leave a previously occupied table; resets to `FREE` after staff attendance.
2. **Spatial Noise Filtering & Nearest-Table Assignment:**
   * Auto-expands calibrated table polygons to cover surrounding seating areas.
   * Filters out empty chairs/furniture false positives via bounding-box geometric constraints and assigns each tracked person exclusively to their nearest table centroid.
3. **Automated Operational Alerts:**
   * **Cleaning Alert:** Instant notification when a table transitions to `NEEDS_CLEANING`.
   * **Unattended Guest Alert:** Flags occupied tables exceeding the wait threshold without a new tracker ID (waiter) approaching.
4. **Queue & Turnover Estimator:** Ranks occupied tables by longest dwell time to predict the next available table.
5. **Peak-Hour Analytics & Staffing Recommendations:** Visualizes hourly occupancy rates alongside actionable shift staffing insights.

---

## 🛠️ Tech Stack

* **Object Detection & Tracking:** Ultralytics YOLOv8 + ByteTrack
* **Spatial Zone Logic:** Supervision (`PolygonZone`) & OpenCV
* **Interactive Dashboard:** Streamlit & Pandas

---

## 📂 Project Structure

```text
smart-table-monitor/
├── assets/
│   └── cctv_sample.mp4      # Sample overhead restaurant CCTV footage
├── config/
│   └── zones.json           # Calibrated polygon coordinates per table
├── src/
│   ├── calibrate_zones.py   # Interactive mouse-click zone calibration tool
│   └── detector.py          # YOLOv8 + ByteTrack + 3-state machine engine
├── app.py                   # Streamlit real-time operations dashboard
├── requirements.txt         # Python dependencies
└── README.md