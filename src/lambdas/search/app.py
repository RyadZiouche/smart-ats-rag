import os
import json
import uuid
import boto3
from pinecone import Pinecone

bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')


# ─────────────────────────────────────────────
# 1. EMBEDDING
# ─────────────────────────────────────────────

def generate_embedding(text: str) -> list:
    """Génère le vecteur pour la requête utilisateur."""
    body = json.dumps({"inputText": text})
    response = bedrock_client.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body,
        accept='application/json',
        contentType='application/json'
    )
    return json.loads(response.get('body').read())['embedding']


# ─────────────────────────────────────────────
# 2. RECHERCHE PINECONE
# ─────────────────────────────────────────────

def search_pinecone(vector: list, top_k: int = 5) -> list:
    """Interroge Pinecone et retourne les chunks les plus proches."""
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    return result['matches']


# ─────────────────────────────────────────────
# 3. GÉNÉRATION RAG (Claude Haiku)
# ─────────────────────────────────────────────

def generate_rag_answer(query: str, chunks: list, conversation_history: list) -> str:
    """
    Construit le prompt RAG avec les chunks récupérés et l'historique de conversation,
    puis appelle Claude Haiku pour générer une réponse sourcée.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks):
        meeting_id = chunk['metadata'].get('meeting_id', 'inconnu')
        chunk_idx = chunk['metadata'].get('chunk_index', '?')
        text = chunk['metadata'].get('text', '')
        context_blocks.append(
            f"[Source {i+1} — Réunion: {meeting_id}, Segment: {chunk_idx}]\n{text}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = """Tu es un assistant intelligent qui répond aux questions sur des réunions.
Tu dois répondre en te basant UNIQUEMENT sur les extraits de transcription fournis.
Cite toujours la source (réunion et numéro de segment) de chaque information.
Si la réponse n'est pas dans les extraits, dis-le clairement."""

    messages = conversation_history.copy()
    messages.append({
        "role": "user",
        "content": f"""Extraits de transcription pertinents :
<context>
{context}
</context>

Question : {query}"""
    })

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": system_prompt,
        "messages": messages
    })

    response = bedrock_client.invoke_model(
        modelId='anthropic.claude-haiku-4-5-20251001-v1:0',
        body=body
    )
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']


# ─────────────────────────────────────────────
# 4. MÉTRIQUES CLOUDWATCH
# ─────────────────────────────────────────────

def push_search_metrics(latency_ms: float, top_score: float):
    """Pousse les métriques de recherche vers CloudWatch."""
    try:
        cloudwatch_client.put_metric_data(
            Namespace='SmartMeetingRAG',
            MetricData=[
                {
                    'MetricName': 'SearchLatencyMs',
                    'Value': latency_ms,
                    'Unit': 'Milliseconds'
                },
                {
                    'MetricName': 'TopChunkSimilarity',
                    'Value': top_score,
                    'Unit': 'None'
                },
                {
                    'MetricName': 'SearchCount',
                    'Value': 1,
                    'Unit': 'Count'
                }
            ]
        )
    except Exception as e:
        # Ne pas bloquer la réponse si CloudWatch échoue
        print(f"Warning CloudWatch : {str(e)}")


# ─────────────────────────────────────────────
# 5. HANDLER LAMBDA
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Déclencheur : POST /search via API Gateway.
    Body attendu : { "query": "...", "conversation_history": [...] }
    Retourne : { "query_id", "answer", "sources", "chunks_used" }
    """
    import time
    start_time = time.time()

    try:
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
        query = body.get('query', '').strip()
        conversation_history = body.get('conversation_history', [])

        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Paramètre 'query' manquant."})
            }

        print(f"Recherche : '{query}'")

        # Pipeline RAG
        vector = generate_embedding(query)
        matches = search_pinecone(vector, top_k=5)

        if not matches:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    "query_id": str(uuid.uuid4()),
                    "answer": "Aucun extrait pertinent trouvé dans les réunions indexées.",
                    "sources": [],
                    "chunks_used": []
                }, ensure_ascii=False)
            }

        answer = generate_rag_answer(query, matches, conversation_history)

        # Construit la liste des sources pour l'UI et le feedback
        sources = [
            {
                "meeting_id": m['metadata'].get('meeting_id'),
                "chunk_index": m['metadata'].get('chunk_index'),
                "score": round(m['score'] * 100, 2),
                "text_preview": m['metadata'].get('text', '')[:200] + "..."
            }
            for m in matches
        ]
        chunk_ids_used = [
            f"{m['metadata'].get('meeting_id')}_chunk_{m['metadata'].get('chunk_index')}"
            for m in matches
        ]

        query_id = str(uuid.uuid4())
        latency_ms = (time.time() - start_time) * 1000
        top_score = matches[0]['score'] if matches else 0.0

        push_search_metrics(latency_ms, top_score)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                "query_id": query_id,
                "answer": answer,
                "sources": sources,
                "chunks_used": chunk_ids_used
            }, ensure_ascii=False)
        }

    except Exception as e:
        print(f"Erreur : {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }