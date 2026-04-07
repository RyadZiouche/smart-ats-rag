import sys
import os
import json
from pypdf import PdfReader
from dotenv import load_dotenv

# Charge les clés secrètes depuis le fichier .env dans l'environnement local
load_dotenv()

# Permet d'importer notre code Lambda depuis le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lambdas.ingestion.app import anonymize_and_extract_skills, generate_embedding, store_in_pinecone
def test_local_extraction():
    # 1. Chemin vers ton PDF de test (chemin robuste, relatif au dossier `tests`)
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_cvs", "cv_test.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Erreur : Le fichier {pdf_path} n'existe pas. Veuillez placer un CV PDF à cet emplacement.")
        return

    print(f"Lecture du fichier local : {pdf_path}")
    
    # Extraction du texte brut
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"
        
    print("Envoi du texte à AWS Bedrock (Claude 3) pour analyse et anonymisation...")
    
    # 2. Anonymisation et structuration
    structured_cv = anonymize_and_extract_skills(raw_text)
    
    print("\nResultat (JSON structuré) :")
    print(json.dumps(structured_cv, indent=2, ensure_ascii=False))

    # 3. Vectorisation
    print("\nGeneration de l'embedding (AWS Titan)...")
    text_to_embed = json.dumps(structured_cv, ensure_ascii=False)
    vector = generate_embedding(text_to_embed)
    print(f"Vecteur genere avec succes (Taille: {len(vector)} dimensions).")

    # 4. Stockage dans Pinecone
    print("\nEnvoi des donnees vers Pinecone...")
    store_in_pinecone(cv_id="cv_test.pdf", vector=vector, metadata=structured_cv)

if __name__ == "__main__":
    test_local_extraction()