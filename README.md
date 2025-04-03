# Analyse des interactions opérateur-machine

Ce projet analyse en temps réel les mouvements d'un opérateur à partir d'une vidéo.

## 🔍 Fonctionnalités

- 🎥 Lecture vidéo avec affichage des poses (MediaPipe)
- 📊 Graphique en temps réel des angles des articulations
- 📋 Tableau dynamique des risques TMS par partie du corps
- ⏯️ Contrôle complet : Lancer, Pause, Réinitialiser
- 💾 Prêt pour Git LFS pour les vidéos lourdes

## 🛠️ Technologies

- Python
- Streamlit
- OpenCV
- MediaPipe
- Matplotlib
- Pandas

## ▶️ Lancer le projet

### 🔁 Si vous venez de cloner le projet ou changez d'ordinateur :

```bash
# 1. Créer un environnement virtuel
python3 -m venv venv

# 2. Activer l'environnement (macOS/Linux)
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application Streamlit
streamlit run scripts/streamlit_app.py
```

### 🚀 Si vous avez déjà l'environnement en local :

```bash
source venv/bin/activate
streamlit run scripts/streamlit_app.py
```

## 📁 Arborescence

```
interaction-operateur-machine/
│
├── data/                  # Contient la vidéo à analyser
├── scripts/
│   └── streamlit_app.py   # Script principal
├── .gitignore
├── README.md
└── requirements.txt
```

## 📹 Remarque

Les vidéos lourdes sont gérées avec [Git LFS](https://git-lfs.github.com/). Veillez à l'installer si vous manipulez les vidéos.

## ❗ Ne pas pousser

Le dossier `venv/` est exclu du dépôt via `.gitignore` car il est propre à votre machine. Inutile de le versionner.