import os
import json
import time
import urllib.parse
from io import BytesIO

import boto3
from pinecone import Pinecone

# Clients AWS
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe', region_name='us-east-1')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

CHUNK_SIZE = 300      # mots par chunk
CHUNK_OVERLAP = 50    # mots de chevauchement entre chunks


# ─────────────────────────────────────────────
# 1. TRANSCRIPTION AUDIO
# ─────────────────────────────────────────────

def transcribe_audio(bucket: str, key: str, job_name: str) -> str:
    """Lance un job AWS Transcribe sur le fichier audio S3 et retourne le texte."""
    s3_uri = f"s3://{bucket}/{key}"
    media_format = key.split('.')[-1].lower()  # mp3, mp4, wav...

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat=media_format,
        LanguageCode='fr-FR',   # adapter si besoin : 'en-US'
        OutputBucketName=bucket,
        OutputKey=f"transcriptions/{job_name}.json"
    )
    print(f"Job Transcribe lancé : {job_name}")

    # Polling jusqu'à la fin du job (max 10 min)
    for _ in range(60):
        status = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )['TranscriptionJob']['TranscriptionJobStatus']

        if status == 'COMPLETED':
            print("Transcription terminée.")
            break
        elif status == 'FAILED':
            raise RuntimeError(f"Job Transcribe échoué : {job_name}")
        
        time.sleep(10)
    else:
        raise TimeoutError("Transcribe : délai dépassé (10 min).")

    # Récupère le fichier JSON de transcription depuis S3
    result = s3_client.get_object(
        Bucket=bucket,
        Key=f"transcriptions/{job_name}.json"
    )
    transcript_data = json.loads(result['Body'].read())
    return transcript_data['results']['transcripts'][0]['transcript']


# ─────────────────────────────────────────────
# 2. CHUNKING DU TEXTE
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Découpe le texte en chunks de ~chunk_size mots avec chevauchement.
    Retourne une liste de dicts avec index et texte de chaque chunk.
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
            "word_end": min(end, len(words))
        })
        start += chunk_size - overlap
        index += 1

    print(f"{len(chunks)} chunks créés depuis la transcription.")
    return chunks


# ─────────────────────────────────────────────
# 3. EMBEDDING BEDROCK
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 4. STOCKAGE PINECONE
# ─────────────────────────────────────────────

def store_chunks_in_pinecone(meeting_id: str, chunks: list[dict]) -> int:
    """Vectorise et stocke chaque chunk dans Pinecone. Retourne le nb de chunks stockés."""
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise EnvironmentError("PINECONE_API_KEY ou PINECONE_INDEX_NAME manquants.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    vectors_to_upsert = []

    for chunk in chunks:
        vector = generate_embedding(chunk["text"])
        chunk_id = f"{meeting_id}_chunk_{chunk['chunk_index']}"

        vectors_to_upsert.append({
            "id": chunk_id,
            "values": vector,
            "metadata": {
                "meeting_id": meeting_id,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "word_start": chunk["word_start"],
                "word_end": chunk["word_end"]
            }
        })

    # Upsert par batch de 100 (limite Pinecone)
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"Succès : {len(vectors_to_upsert)} chunks indexés pour la réunion {meeting_id}.")
    return len(vectors_to_upsert)


# ─────────────────────────────────────────────
# 5. HANDLER LAMBDA
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Déclencheur : s3:ObjectCreated sur le bucket audio.
    Pipeline : audio MP3/WAV → Transcribe → chunking → Titan embed → Pinecone
    """
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(
            event['Records'][0]['s3']['object']['key'], encoding='utf-8'
        )

        print(f"Nouveau fichier détecté : s3://{bucket}/{key}")

        # Identifiant unique de réunion basé sur le nom de fichier
        meeting_id = os.path.splitext(os.path.basename(key))[0]
        job_name = f"meeting-{meeting_id}-{int(time.time())}"

        # Pipeline complet
        transcript_text = transcribe_audio(bucket, key, job_name)
        chunks = chunk_text(transcript_text)
        nb_chunks = store_chunks_in_pinecone(meeting_id, chunks)

        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": f"Réunion '{meeting_id}' indexée avec succès.",
                "chunks_created": nb_chunks
            })
        }

    except Exception as e:
        print(f"Échec du pipeline d'ingestion : {str(e)}")
        raise e