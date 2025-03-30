from flask import Flask, render_template, request, jsonify
from extractz import process_pdf
import os
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_script():
    print("Le bouton a bien ete clique !")
    try:
        if 'pdf_file' not in request.files:
            return jsonify({"error": "Aucun fichier PDF fourni"}), 400

        pdf = request.files['pdf_file']
        if pdf.filename == '':
            return jsonify({"error": "Nom de fichier invalide"}), 400

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
        pdf.save(pdf_path)

        print("=> Lancement de process_pdf() avec:", pdf_path)
        output_json = "structured_extracted_data.json"
        extracted_data = process_pdf(pdf_path, "output_images", output_json)
        print("Traitement termine")

        return jsonify({
            "message": "Traitement termine avec succes !",
            "data": extracted_data
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Serveur en cours de demarrage...")
    app.run(debug=True)
