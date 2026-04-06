"""
Smart ATS - Ingestion & Anonymization Lambda
--------------------------------------------
Triggered by S3 events, this function extracts text from uploaded resumes (PDFs),
uses AWS Bedrock (Claude 3) to extract structured JSON and anonymize PII (GDPR compliance),
generates text embeddings (Titan), and stores the vector in Pinecone for RAG retrieval.
"""

import os
import json
import urllib.parse
from io import BytesIO

import boto3
from pypdf import PdfReader
from pinecone import Pinecone

# AWS Clients initialization
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

def extract_text_from_pdf(bucket: str, key: str) -> str:
    """
    Downloads a PDF from S3 into memory and extracts raw text.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    pdf_file = BytesIO(response['Body'].read())
    reader = PdfReader(pdf_file)
    
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def anonymize_and_extract_skills(cv_text: str) -> dict:
    """
    Leverages Claude 3 Haiku to parse the raw resume, strip PII (GDPR),
    and return a structured JSON object representing the candidate's profile.
    """
    prompt = f"""Tu es un expert RH et un système de parsing de CV très strict.
Voici le texte brut extrait d'un CV.
Ta mission est de l'analyser, de supprimer TOUTES les données personnelles (Nom, Prénom, Email, Téléphone, Adresse), et d'extraire les informations professionnelles au format JSON strict.

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
        "temperature": 0.0, # Zero temperature to ensure deterministic JSON output and prevent hallucinations
        "messages": [{"role": "user", "content": prompt}]
    })

    response = bedrock_client.invoke_model(
        modelId='anthropic.claude-3-haiku-20240307-v1:0',
        body=body
    )
    
    response_body = json.loads(response.get('body').read())
    return json.loads(response_body['content'][0]['text'])

def generate_embedding(text: str) -> list:
    """
    Calls Amazon Titan Text Embeddings to vectorize the JSON profile.
    """
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
    """
    Upserts the generated vector and candidate metadata into the Pinecone Vector DB.
    Requires PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Warning: Pinecone credentials missing. Skipping vector storage (Local test mode).")
        return
        
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    index.upsert(
        vectors=[
            {
                "id": cv_id, 
                "values": vector, 
                "metadata": metadata
            }
        ]
    )
    print(f"Vector successfully stored in Pinecone for CV ID: {cv_id}")

def lambda_handler(event, context):
    """
    AWS Lambda entry point. Triggered by S3 PutObject events.
    """
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        print(f"Starting ingestion pipeline for: {key} in bucket: {bucket}")

        # 1. Document Parsing
        raw_text = extract_text_from_pdf(bucket, key)
        
        # 2. PII Stripping & JSON Extraction (LLM)
        structured_cv = anonymize_and_extract_skills(raw_text)
        
        # 3. Vectorization (Embeddings)
        text_to_embed = json.dumps(structured_cv, ensure_ascii=False)
        vector = generate_embedding(text_to_embed)
        
        # 4. Vector Storage
        store_in_pinecone(cv_id=key, vector=vector, metadata=structured_cv)
        
        return {
            'statusCode': 200,
            'body': json.dumps('Ingestion pipeline completed successfully.')
        }

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise e