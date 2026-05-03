import os
import json
import urllib.parse
from io import BytesIO

import boto3
from pypdf import PdfReader
from pinecone import Pinecone

# Initialisation des clients 
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

def extract_text_from_pdf(bucket: str, key: str) -> str:
    """Télécharge le PDF depuis S3 et extrait le texte brut."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    pdf_file = BytesIO(response['Body'].read())
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def anonymize_and_extract_skills(cv_text: str) -> dict:
    """Utilise Claude 4.5 Haiku pour anonymiser (RGPD) et structurer le CV."""
    prompt = f"""Tu es un expert RH et un système de parsing de CV très strict.
Voici le texte brut extrait d'un CV.
Ta mission est de l'analyser, de supprimer TOUTES les données personnelles (Nom, Prénom, Email, Téléphone, Adresse), et d'extraire les informations professionnelles au format JSON strict.

RETOURNE UNIQUEMENT LE JSON. AUCUN TEXTE AVANT OU APRÈS.

CV Brut :
<cv>
{cv_text}
</cv>

Réponds UNIQUEMENT avec un objet JSON valide ayant cette structure exacte :
{{
    "titre_profil": "...",
    "annees_experience_total": 0,
    "competences_techniques": ["...", "..."],
    "soft_skills": ["...", "..."],
    "experiences_cles": ["...", "..."]
}}"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = bedrock_client.invoke_model(
        modelId='us.anthropic.claude-haiku-4-5-20251001-v1:0',
        body=body
    )
    
    response_body = json.loads(response.get('body').read())
    res_text = response_body['content'][0]['text']
    
    start = res_text.find('{')
    end = res_text.rfind('}') + 1
    
    if start == -1 or end == 0:
        raise ValueError("Le modèle n'a généré aucun format JSON valide.")
        
    json_clean = res_text[start:end]
    return json.loads(json_clean)

def generate_embedding(text: str) -> list:
    """Génère un vecteur de 1536 dimensions avec Amazon Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock_client.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body,
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response.get('body').read())
    return response_body['embedding']

def store_in_pinecone(cv_id: str, vector: list, metadata: dict) -> None:
    """Enregistre le vecteur et les métadonnées dans Pinecone."""
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Erreur : Clés Pinecone absentes des variables d'environnement.")
        return
        
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    index.upsert(
        vectors=[{
            "id": cv_id, 
            "values": vector, 
            "metadata": metadata
        }]
    )
    print(f"Succès : CV {cv_id} vectorisé et stocké dans Pinecone.")

def lambda_handler(event, context):
    """Point d'entrée AWS Lambda déclenché par S3."""
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        print(f" Début du traitement pour : {key}")
        raw_text = extract_text_from_pdf(bucket, key)
        structured_cv = anonymize_and_extract_skills(raw_text)
        text_to_embed = json.dumps(structured_cv, ensure_ascii=False)
        vector = generate_embedding(text_to_embed)
        store_in_pinecone(cv_id=key, vector=vector, metadata=structured_cv)
        
        return {'statusCode': 200, 'body': json.dumps(f"Le CV {key} a été traité avec succès !")}
    except Exception as e:
        print(f" Échec du pipeline : {str(e)}")
        raise e