import os
import json
import boto3
from pinecone import Pinecone

# On initialise le client Bedrock pour Amazon Titan
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

def generate_embedding(text: str) -> list:
    """Génère le vecteur pour la requête du recruteur."""
    body = json.dumps({"inputText": text})
    response = bedrock_client.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body,
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response.get('body').read())
    return response_body['embedding']

def lambda_handler(event, context):
    """Point d'entrée AWS Lambda déclenché par API Gateway."""
    try:
        if 'body' in event and isinstance(event['body'], str):
            body = json.loads(event['body'])
            query = body.get('query', '')
        else:
            query = event.get('query', '')

        if not query:
            return {'statusCode': 400, 'body': json.dumps("Erreur: Aucune requête fournie.")}

        print(f"🔍 Recherche en cours pour : '{query}'")
        vector = generate_embedding(query)

        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        search_result = index.query(
            vector=vector,
            top_k=3,
            include_metadata=True
        )

        matches = []
        for match in search_result['matches']:
            matches.append({
                "fichier_cv": match['id'],
                "score_pertinence": round(match['score'] * 100, 2),
                "donnees": match['metadata']
            })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({"resultats": matches}, ensure_ascii=False)
        }

    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f"Erreur interne : {str(e)}")}