import streamlit as st
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp

st.set_page_config(page_title="Analyse TMS", layout="wide")
st.title("Analyse des interactions opérateur-machine 🦾")

# === Fonction angle ===
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    angle = math.acos(dot / (mag_ba * mag_bc + 1e-6))
    return math.degrees(angle)

# === Initialisation session_state ===
def init_state():
    st.session_state.frame_id = 0
    st.session_state.paused = True
    st.session_state.launched = False
    st.session_state.angle_data = {k: [] for k in ["Coude_droit", "Bras_droit", "Epaule_droite", "Dos", "Cou"]}
    st.session_state.action_counts = {k: 0 for k in st.session_state.angle_data}
    st.session_state.previous_angles = {k: None for k in st.session_state.angle_data}
    st.session_state.cooldowns = {k: 0 for k in st.session_state.angle_data}

if "frame_id" not in st.session_state:
    init_state()

# === MediaPipe Pose ===
if "pose" not in st.session_state:
    st.session_state.pose = mp.solutions.pose.Pose()
pose = st.session_state.pose

# === Vidéo ===
video_path = "data/VideoPresse.mp4"
if not os.path.exists(video_path):
    st.error("Vidéo introuvable")
    st.stop()

# === Interface boutons ===
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("🔁 Réinitialiser"):
        init_state()
        st.rerun()
with b2:
    if st.button("▶ Lancer", disabled=st.session_state.launched):
        st.session_state.paused = False
        st.session_state.launched = True
with b3:
    if st.session_state.launched:
        label = "⏸ Pause" if not st.session_state.paused else "▶ Reprendre"
        if st.button(label):
            st.session_state.paused = not st.session_state.paused

# === Layout vidéo + graph + tableau ===
c1, c2 = st.columns([1, 2])
with c1:
    video_slot = st.empty()
with c2:
    graph_slot = st.empty()
    table_slot = st.empty()

# === Afficher le premier frame si rien n’est lancé ===
if not st.session_state.launched:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 7)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_slot.image(frame_rgb, channels="RGB", caption="Vidéo en pause")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Évolution des angles")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (°)")
    ax.grid(True)
    graph_slot.pyplot(fig)

    table_slot.dataframe(pd.DataFrame(columns=["Partie", "Mouvements", "Pic critique (°)", "Risque TMS"]), use_container_width=True)

# === Si vidéo lancée mais en pause, afficher l'état actuel sans avancer ===
if st.session_state.launched and st.session_state.paused:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 7 + st.session_state.frame_id)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_slot.image(frame_rgb, channels="RGB", caption=f"Frame {int(st.session_state.frame_id)} (pause)")

    fig, ax = plt.subplots(figsize=(8, 4))
    for part in st.session_state.angle_data:
        ax.plot(st.session_state.angle_data[part], label=part)
    ax.set_title("Évolution des angles")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (°)")
    ax.grid(True)
    ax.legend()
    graph_slot.pyplot(fig)

    summary = []
    risk_map = {"Faible": 1, "Modéré": 2, "Élevé": 3}
    score_total, score_machine, all_peaks = 0, 0, []

    for part in st.session_state.angle_data:
        angles = st.session_state.angle_data[part]
        moves = st.session_state.action_counts[part]
        peak = round(max(abs(max(angles) - 180), abs(min(angles) - 0)), 2) if angles else 0
        all_peaks.append(peak)
        if moves <= 10: risk = "Faible"
        elif moves <= 20: risk = "Modéré"
        else: risk = "Élevé"
        summary.append({"Partie": part, "Mouvements": moves, "Pic critique (°)": peak, "Risque TMS": risk})
        score_total += risk_map[risk]
        if part in ["Coude_droit", "Bras_droit", "Epaule_droite"]:
            score_machine += risk_map[risk]

    moy_humain = score_total / len(st.session_state.angle_data)
    moy_machine = score_machine / 3
    summary.append({"Partie": "TOTAL", "Mouvements": sum(st.session_state.action_counts.values()),
                    "Pic critique (°)": max(all_peaks),
                    "Risque TMS": f"Humain: {'Faible' if moy_humain<1.5 else 'Modéré' if moy_humain<2.5 else 'Élevé'} | "
                                   f"Machine: {'Faible' if moy_machine<1.5 else 'Modéré' if moy_machine<2.5 else 'Élevé'}"})

    table_slot.dataframe(pd.DataFrame(summary), use_container_width=True)

# === Lecture vidéo fluide avec analyse en direct ===
if st.session_state.launched and not st.session_state.paused:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 7 + st.session_state.frame_id)

    while cap.isOpened() and not st.session_state.paused:
        ret, frame = cap.read()
        if not ret or st.session_state.frame_id >= total_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            def pt(p): return (lm[p].x, lm[p].y)

            parts = {
                "Coude_droit": (pt(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER), pt(mp.solutions.pose.PoseLandmark.RIGHT_ELBOW), pt(mp.solutions.pose.PoseLandmark.RIGHT_WRIST)),
                "Bras_droit": (pt(mp.solutions.pose.PoseLandmark.RIGHT_ELBOW), pt(mp.solutions.pose.PoseLandmark.RIGHT_WRIST), pt(mp.solutions.pose.PoseLandmark.RIGHT_INDEX)),
                "Epaule_droite": (pt(mp.solutions.pose.PoseLandmark.RIGHT_HIP), pt(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER), pt(mp.solutions.pose.PoseLandmark.RIGHT_ELBOW)),
                "Dos": (pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER), pt(mp.solutions.pose.PoseLandmark.RIGHT_HIP), pt(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)),
                "Cou": (pt(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER), pt(mp.solutions.pose.PoseLandmark.NOSE), pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER))
            }

            thresholds = {"Coude_droit": 20, "Bras_droit": 20, "Epaule_droite": 15, "Dos": 10, "Cou": 15}
            for part, (a, b, c) in parts.items():
                angle = calculate_angle(a, b, c)
                st.session_state.angle_data[part].append(angle)
                prev = st.session_state.previous_angles[part]
                if prev is not None and abs(angle - prev) > thresholds[part] and st.session_state.cooldowns[part] == 0:
                    st.session_state.action_counts[part] += 1
                    st.session_state.cooldowns[part] = 5
                st.session_state.previous_angles[part] = angle
                if st.session_state.cooldowns[part] > 0:
                    st.session_state.cooldowns[part] -= 1

            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {int(st.session_state.frame_id)}")

        fig, ax = plt.subplots(figsize=(8, 4))
        for part in st.session_state.angle_data:
            ax.plot(st.session_state.angle_data[part], label=part)
        ax.set_title("Évolution des angles")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.grid(True)
        ax.legend()
        graph_slot.pyplot(fig)

        summary = []
        risk_map = {"Faible": 1, "Modéré": 2, "Élevé": 3}
        score_total, score_machine, all_peaks = 0, 0, []

        for part in st.session_state.angle_data:
            angles = st.session_state.angle_data[part]
            moves = st.session_state.action_counts[part]
            peak = round(max(abs(max(angles) - 180), abs(min(angles) - 0)), 2) if angles else 0
            all_peaks.append(peak)
            if moves <= 10: risk = "Faible"
            elif moves <= 20: risk = "Modéré"
            else: risk = "Élevé"
            summary.append({"Partie": part, "Mouvements": moves, "Pic critique (°)": peak, "Risque TMS": risk})
            score_total += risk_map[risk]
            if part in ["Coude_droit", "Bras_droit", "Epaule_droite"]:
                score_machine += risk_map[risk]

        moy_humain = score_total / len(st.session_state.angle_data)
        moy_machine = score_machine / 3
        summary.append({"Partie": "TOTAL", "Mouvements": sum(st.session_state.action_counts.values()),
                        "Pic critique (°)": max(all_peaks),
                        "Risque TMS": f"Humain: {'Faible' if moy_humain<1.5 else 'Modéré' if moy_humain<2.5 else 'Élevé'} | "
                                       f"Machine: {'Faible' if moy_machine<1.5 else 'Modéré' if moy_machine<2.5 else 'Élevé'}"})

        table_slot.dataframe(pd.DataFrame(summary), use_container_width=True)

        st.session_state.frame_id += 1
        time.sleep(1 / fps)

    cap.release()