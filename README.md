# Interaction Opérateur - Machine (Rework)

Ce projet vise à détecter automatiquement les mouvements techniques d’un opérateur manipulant une machine à partir d’une vidéo.

Grâce à la détection de pose (MediaPipe), les mouvements des bras, épaules, coudes et du dos seront analysés afin :
- D’identifier les risques ergonomiques (TMS)
- De mesurer l’impact potentiel sur la dégradation de la machine
- Et d’afficher ces informations dans une interface simple et claire

## Arborescence
- `data/` : vidéos sources et données extraites
- `scripts/` : traitements Python (analyse vidéo)
- `interface/` : affichage des résultats
- `results/` : exports visuels ou fichiers
- `requirements.txt` : dépendances Python
- `.gitignore` : fichiers à ne pas versionner

## Technologies prévues
- Python
- MediaPipe
- OpenCV
- Interface Web (HTML/CSS ou Streamlit)

## Auteur
Mohamed Ali KAMMOUN – Université de Lorraine
