# Importation des modules nécessaires
from flask import Flask, render_template, request, jsonify  # Flask pour le serveur web
from extractz import process_pdf                           # Fonction de traitement OCR personnalisée
import os                                                  # Gestion des fichiers/systèmes
import traceback                                           # Pour afficher les erreurs détaillées en cas d'exception

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration du dossier d'upload des fichiers
app.config['UPLOAD_FOLDER'] = 'uploads'

# Création du dossier d'upload s’il n'existe pas
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Route de la page d'accueil (GET uniquement)
@app.route('/')
def index():
    return render_template('index.html')  # Affiche le template HTML (formulaire d’upload par exemple)

# Route déclenchée après envoi de formulaire (POST)
@app.route('/run', methods=['POST'])
def run_script():
    print("Le bouton a bien ete clique !")
    try:
        # Vérifie si un fichier PDF a été envoyé dans la requête
        if 'pdf_file' not in request.files:
            return jsonify({"error": "Aucun fichier PDF fourni"}), 400

        pdf = request.files['pdf_file']
        # Vérifie que le nom du fichier n'est pas vide
        if pdf.filename == '':
            return jsonify({"error": "Nom de fichier invalide"}), 400

        # Sauvegarde le fichier dans le dossier d'uploads
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
        pdf.save(pdf_path)

        print("=> Lancement de process_pdf() avec:", pdf_path)

        # Lance le traitement du fichier PDF avec OCR
        output_json = "structured_extracted_data.json"
        extracted_data = process_pdf(pdf_path, "output_images", output_json)

        print("Traitement termine")

        # Retourne les données extraites en JSON
        return jsonify({
            "message": "Traitement termine avec succes !",
            "data": extracted_data
        })

    # En cas d’erreur, capture l’exception et retourne une réponse 500 avec trace
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Point d’entrée du serveur Flask
if __name__ == '__main__':
    print("Serveur en cours de demarrage...")
    app.run(debug=True)  # Lance le serveur en mode debug (affiche les erreurs dans la console)
