# CYScan - Extraction intelligente de tableaux depuis des PDF

CYScan est une application web intuitive et élégante, développée par des étudiants de CY Tech, qui permet de détecter, extraire et structurer automatiquement des tableaux historiques présents dans des fichiers PDF. Elle utilise les technologies de reconnaissance optique de caractères (OCR) les plus avancées pour transformer des documents complexes en données exploitables.

---

## 🎯 Objectif du projet
Développer un outil automatisé capable de détecter des tableaux présents dans des documents PDF historiques (par exemple, des tables de mortalité ou de causes de décès) et de les restituer sous forme structurée pour leur réutilisation scientifique ou institutionnelle.

---

## 💻 Technologies utilisées

- **Python** (traitement principal)
- **Flask** (interface web)
- **PaddleOCR** (reconnaissance optique de caractères)
- **OpenCV** (traitement d’image)
- **pdf2image** (conversion de PDF en images)
- **HTML / CSS / JavaScript** (interface moderne et responsive)

---

## 📁 Architecture du projet

```
CYScan/
├── app.py                  # Backend Flask
├── extractz.py             # Traitement OCR et extraction des tableaux
├── templates/
│   └── index.html          # Interface utilisateur
├── uploads/                # Dossier de téléversement PDF
├── output_images/          # Images extraites des PDF
├── structured_extracted_data.json  # Résultat final JSON
├── README.md               # Documentation
```

---

## 🚀 Installation & exécution

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-utilisateur/CYScan.git
cd CYScan
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

Assurez-vous d’avoir installé **Poppler** et de bien configurer le chemin dans `extractz.py` :
```python
os.environ["PATH"] += os.pathsep + r"H:\\poppler-24.08.0\\Library\\bin"
```

### 3. Lancer l’application
```bash
python app.py
```
Puis ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000) dans votre navigateur.

---

## 🧪 Fonctionnement
1. L’utilisateur téléverse un fichier PDF via l’interface web.
2. Le backend analyse chaque page, détecte les tableaux via OpenCV.
3. Chaque tableau est extrait en image puis analysé avec PaddleOCR.
4. Les données sont nettoyées et formatées dans un fichier JSON lisible.

---

## 👨‍👩‍👧‍👦 Membres de l'équipe
- **BAIROUKI Yssam**
- **BELMAHI Zakarya**
- **NOLLAND Léo**
- **MCHICH Nada**
- **CHAOUCHE-VIGNOLLES Rayan**

Projet universitaire réalisé dans le cadre du semestre CY Tech 2025.

---

## 📄 Licence
Ce projet est destiné à un usage éducatif uniquement.

