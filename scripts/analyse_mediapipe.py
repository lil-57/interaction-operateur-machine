import cv2
import mediapipe as mp
import math
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

# === Paramètres ===
angle_thresholds = {
    "Coude_droit": 20,
    "Bras_droit": 20,
    "Epaule_droite": 15,
    "Dos": 10,
    "Cou": 15
}
cooldown_frames = 5
cooldowns = {key: 0 for key in angle_thresholds}
previous_angles = {key: None for key in angle_thresholds}
angle_data = {key: [] for key in angle_thresholds}
action_counts = {key: 0 for key in angle_thresholds}
frame_id = 0

# === Fonctions utiles ===
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    angle_rad = math.acos(dot_product / (mag_ba * mag_bc + 1e-6))
    return math.degrees(angle_rad)

# === MediaPipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("data/VideoPresse.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 7)

# Créer dossier results
os.makedirs("results", exist_ok=True)
csv_writer = csv.writer(open("results/actions_completes.csv", 'w', newline=''))
csv_writer.writerow(["Frame"] + list(angle_thresholds.keys()) + ["Action_detectee"])

# === Analyse frame par frame ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    actions_detected = []
    angles_this_frame = {}

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        def get_point(part):
            p = lm[part]
            return (p.x, p.y)

        if lm[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
            points = {
                "Coude_droit": (get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                get_point(mp_pose.PoseLandmark.RIGHT_ELBOW),
                                get_point(mp_pose.PoseLandmark.RIGHT_WRIST)),
                "Bras_droit": (get_point(mp_pose.PoseLandmark.RIGHT_ELBOW),
                               get_point(mp_pose.PoseLandmark.RIGHT_WRIST),
                               get_point(mp_pose.PoseLandmark.RIGHT_INDEX)),
                "Epaule_droite": (get_point(mp_pose.PoseLandmark.RIGHT_HIP),
                                  get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                  get_point(mp_pose.PoseLandmark.RIGHT_ELBOW)),
                "Dos": (get_point(mp_pose.PoseLandmark.LEFT_SHOULDER),
                        get_point(mp_pose.PoseLandmark.RIGHT_HIP),
                        get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)),
                "Cou": (get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                        get_point(mp_pose.PoseLandmark.NOSE),
                        get_point(mp_pose.PoseLandmark.LEFT_SHOULDER))
            }

            for part, (a, b, c) in points.items():
                angle = calculate_angle(a, b, c)
                angles_this_frame[part] = angle
                angle_data[part].append(angle)

                if previous_angles[part] is not None:
                    if cooldowns[part] == 0 and abs(angle - previous_angles[part]) > angle_thresholds[part]:
                        action_counts[part] += 1
                        cooldowns[part] = cooldown_frames

                previous_angles[part] = angle

            # Affichage des landmarks avec connexions
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    for part in cooldowns:
        if cooldowns[part] > 0:
            cooldowns[part] -= 1

    # Sauvegarde CSV
    row = [frame_id] + [round(angles_this_frame.get(k, 0), 2) for k in angle_thresholds.keys()] + [sum(action_counts.values())]
    csv_writer.writerow(row)
    frame_id += 1

    # Affichage simple
    cv2.putText(frame, f"Actions totales: {sum(action_counts.values())}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Analyse TMS", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Analyse finale ===
summary = []
risque_map = {"Faible": 1, "Modéré": 2, "Élevé": 3}
score_total = 0
score_machine = 0

for part in angle_thresholds:
    angles = angle_data[part]
    mov = action_counts[part]
    pic = round(max(abs(max(angles) - 180), abs(min(angles) - 0)), 2)

    if mov <= 10:
        risk = "Faible"
    elif mov <= 20:
        risk = "Modéré"
    else:
        risk = "Élevé"

    score_total += risque_map[risk]
    if part in ["Coude_droit", "Bras_droit", "Epaule_droite"]:
        score_machine += risque_map[risk]

    summary.append({"Partie": part, "Mouvements": mov, "Pic critique (°)": pic, "Risque TMS": risk})

moy_humain = score_total / len(angle_thresholds)
moy_machine = score_machine / 3

summary.append({"Partie": "TOTAL", "Mouvements": "-", "Pic critique (°)": "-",
                "Risque TMS": f"Humain: {'Faible' if moy_humain<1.5 else 'Modéré' if moy_humain<2.5 else 'Élevé'} | "
                               f"Machine: {'Faible' if moy_machine<1.5 else 'Modéré' if moy_machine<2.5 else 'Élevé'}"})

# === Graphe affiché ===
plt.figure(figsize=(12, 6))
for part in angle_data:
    plt.plot(angle_data[part], label=part)
plt.title("Évolution des angles par partie du corps")
plt.xlabel("Frame")
plt.ylabel("Angle (°)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Tableau affiché ===
print("\nTableau récapitulatif des risques TMS :")
print(tabulate(summary, headers="keys", tablefmt="grid"))

# === Optionnel : enregistrement du graphe et du tableau ===
save = input("\nSouhaitez-vous enregistrer le graphe et le tableau ? (o/n) : ")
if save.lower() == 'o':
    # Sauvegarder le graphe
    plt.figure(figsize=(12, 6))
    for part in angle_data:
        plt.plot(angle_data[part], label=part)
    plt.title("Évolution des angles par partie du corps")
    plt.xlabel("Frame")
    plt.ylabel("Angle (°)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/evolution_angles.png")

    # Sauvegarder le tableau
    pd.DataFrame(summary).to_csv("results/recapitulatif_tms.csv", index=False)
    print("\nFichiers enregistrés dans le dossier 'results/'")
