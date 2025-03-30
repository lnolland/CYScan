import os
import shutil

# Ajoute ici le bon chemin vers Poppler
os.environ["PATH"] += os.pathsep + r"H:\poppler-24.08.0\Library\bin"

print("[Poppler] PATH configuré :", os.environ["PATH"])
print("[Check] pdftoppm trouvé :", shutil.which("pdftoppm"))
