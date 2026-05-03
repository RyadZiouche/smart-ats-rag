import os
import json
import boto3
from boto3.dynamodb.conditions import Attr

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
s3_client = boto3.client('s3')

TABLE_NAME  = os.environ.get("DYNAMODB_TABLE", "FeedbackStore")
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")


# ─────────────────────────────────────────────
# 1. LECTURE DES FEEDBACKS NÉGATIFS
# ─────────────────────────────────────────────

def get_negative_feedbacks() -> list:
    """
    Scan DynamoDB pour récupérer tous les feedbacks négatifs
    non encore traités (reindexed = False).
    """
    table = dynamodb.Table(TABLE_NAME)
    response = table.scan(
        FilterExpression=Attr('score').eq(-1) & Attr('reindexed').eq(False)
    )
    items = response.get('Items', [])
    print(f"{len(items)} feedbacks négatifs non traités trouvés.")
    return items


# ─────────────────────────────────────────────
# 2. RÉCUPÉRATION DU TEXTE ORIGINAL DEPUIS S3
# ─────────────────────────────────────────────

def fetch_transcript_text(meeting_id: str) -> str | None:
    """
    Récupère la transcription brute depuis S3.
    Format attendu : transcriptions/{meeting_id}.json
    """
    try:
        result = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=f"transcriptions/{meeting_id}.json"
        )
        data = json.loads(result['Body'].read())
        return data['results']['transcripts'][0]['transcript']
    except Exception as e:
        print(f"Impossible de récupérer la transcription de '{meeting_id}' : {e}")
        return None


# ─────────────────────────────────────────────
# 3. RE-CHUNKING ADAPTATIF
# ─────────────────────────────────────────────

def rechunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[dict]:
    """
    Re-découpe avec des chunks plus petits (150 mots vs 300 initialement).
    La stratégie plus fine améliore la précision pour les requêtes spécifiques.
    """
    words = text.split()
    chunks = []
    start = 0
    index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "chunk_index": index,
            "text": " ".join(chunk_words),
            "word_start": start,
            "word_end": min(end, len(words)),
            "strategy": "fine_grain_reindex"     # tracé pour l'audit
        })
        start += chunk_size - overlap
        index += 1

    return chunks


# ─────────────────────────────────────────────
# 4. EMBEDDING + UPSERT PINECONE
# ─────────────────────────────────────────────

def generate_embedding(text: str) -> list:
    body = json.dumps({"inputText": text})
    response = bedrock_client.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body,
        accept='application/json',
        contentType='application/json'
    )
    return json.loads(response.get('body').read())['embedding']


def reindex_chunks_in_pinecone(meeting_id: str, chunks: list[dict]) -> int:
    """Ré-embed et upsert les nouveaux chunks dans Pinecone (écrase les anciens)."""
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    vectors = []
    for chunk in chunks:
        vector = generate_embedding(chunk["text"])
        chunk_id = f"{meeting_id}_chunk_{chunk['chunk_index']}"
        vectors.append({
            "id": chunk_id,
            "values": vector,
            "metadata": {
                "meeting_id": meeting_id,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "word_start": chunk["word_start"],
                "word_end": chunk["word_end"],
                "strategy": chunk["strategy"]
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    print(f"{len(vectors)} chunks réindexés pour '{meeting_id}'.")
    return len(vectors)


# ─────────────────────────────────────────────
# 5. MARQUAGE DES FEEDBACKS TRAITÉS
# ─────────────────────────────────────────────

def mark_feedback_as_reindexed(query_id: str, timestamp: str) -> None:
    """Met à jour le flag reindexed = True dans DynamoDB."""
    table = dynamodb.Table(TABLE_NAME)
    table.update_item(
        Key={"query_id": query_id, "timestamp": timestamp},
        UpdateExpression="SET reindexed = :val",
        ExpressionAttributeValues={":val": True}
    )


# ─────────────────────────────────────────────
# 6. MÉTRIQUES CLOUDWATCH
# ─────────────────────────────────────────────

def push_reindex_metrics(nb_meetings: int, nb_chunks: int, nb_feedbacks: int) -> None:
    try:
        cloudwatch_client.put_metric_data(
            Namespace='SmartMeetingRAG',
            MetricData=[
                {'MetricName': 'ReindexedMeetings', 'Value': nb_meetings, 'Unit': 'Count'},
                {'MetricName': 'ReindexedChunks',   'Value': nb_chunks,   'Unit': 'Count'},
                {'MetricName': 'ProcessedNegativeFeedbacks', 'Value': nb_feedbacks, 'Unit': 'Count'}
            ]
        )
    except Exception as e:
        print(f"Warning CloudWatch : {str(e)}")


# ─────────────────────────────────────────────
# 7. HANDLER LAMBDA
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Déclencheur : EventBridge rule, toutes les semaines (cron(0 2 ? * MON *)).
    Pipeline :
      1. Lit les feedbacks négatifs non traités dans DynamoDB
      2. Groupe par meeting_id
      3. Pour chaque réunion → re-fetch transcript S3 → rechunking fin → re-embed → Pinecone
      4. Marque les feedbacks comme traités
      5. Pousse les métriques dans CloudWatch
    """
    print("Démarrage du job de réindexation hebdomadaire.")

    negative_feedbacks = get_negative_feedbacks()

    if not negative_feedbacks:
        print("Aucun feedback négatif à traiter. Job terminé.")
        return {'statusCode': 200, 'body': json.dumps({"message": "Rien à réindexer."})}

    # Groupe les feedbacks par meeting_id pour éviter les re-fetch doublons
    meetings_to_reindex: dict[str, list] = {}
    for fb in negative_feedbacks:
        mid = fb.get('meeting_id', 'unknown')
        meetings_to_reindex.setdefault(mid, []).append(fb)

    total_chunks_reindexed = 0
    meetings_reindexed = 0

    for meeting_id, feedbacks in meetings_to_reindex.items():
        print(f"Réindexation de '{meeting_id}' ({len(feedbacks)} feedbacks négatifs).")

        transcript_text = fetch_transcript_text(meeting_id)
        if not transcript_text:
            print(f"Transcription introuvable pour '{meeting_id}', ignorée.")
            continue

        new_chunks = rechunk_text(transcript_text)
        nb = reindex_chunks_in_pinecone(meeting_id, new_chunks)
        total_chunks_reindexed += nb
        meetings_reindexed += 1

        # Marque tous les feedbacks de cette réunion comme traités
        for fb in feedbacks:
            mark_feedback_as_reindexed(fb['query_id'], fb['timestamp'])

    push_reindex_metrics(
        nb_meetings=meetings_reindexed,
        nb_chunks=total_chunks_reindexed,
        nb_feedbacks=len(negative_feedbacks)
    )

    summary = {
        "meetings_reindexed": meetings_reindexed,
        "total_chunks_reindexed": total_chunks_reindexed,
        "feedbacks_processed": len(negative_feedbacks)
    }
    print(f"Réindexation terminée : {summary}")

    return {'statusCode': 200, 'body': json.dumps(summary)}