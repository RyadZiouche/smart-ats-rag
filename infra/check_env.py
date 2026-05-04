"""
check_env.py
────────────
Vérifie que toutes les variables d'environnement et dépendances
sont correctement configurées avant de déployer les Lambdas.

Usage :
    python infra/check_env.py
"""

import os
import sys
import boto3
import json
from dotenv import load_dotenv

load_dotenv()

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def ok(msg: str) -> None:
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  ✅ {msg}")


def fail(msg: str) -> None:
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  ❌ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


# ─────────────────────────────────────────────
# 1. VARIABLES D'ENVIRONNEMENT
# ─────────────────────────────────────────────

def check_env_vars() -> None:
    print("\n── Variables d'environnement ───────────────────────")

    required = {
        "AWS_REGION":        "Région AWS (ex: us-east-1)",
        "S3_BUCKET_NAME":    "Bucket S3 pour les fichiers audio",
        "PINECONE_API_KEY":  "Clé API Pinecone",
        "PINECONE_INDEX_NAME": "Nom de l'index Pinecone (1536 dims, cosine)",
        "DYNAMODB_TABLE":    "Nom de la table DynamoDB",
        "API_BASE_URL":      "URL de base API Gateway",
    }

    for var, description in required.items():
        value = os.environ.get(var)
        if not value:
            fail(f"{var} manquant — {description}")
        elif value.startswith("your-") or value == "ACCOUNT_ID":
            warn(f"{var} semble être une valeur placeholder : '{value}'")
        else:
            ok(f"{var} = {value[:40]}{'...' if len(value) > 40 else ''}")


# ─────────────────────────────────────────────
# 2. CREDENTIALS AWS
# ─────────────────────────────────────────────

def check_aws_credentials() -> None:
    print("\n── Credentials AWS ─────────────────────────────────")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        ok(f"Account ID : {identity['Account']}")
        ok(f"ARN        : {identity['Arn']}")
    except Exception as e:
        fail(f"Credentials AWS invalides ou absents : {e}")


# ─────────────────────────────────────────────
# 3. BUCKET S3
# ─────────────────────────────────────────────

def check_s3() -> None:
    print("\n── S3 ──────────────────────────────────────────────")
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket:
        fail("S3_BUCKET_NAME non défini, vérification ignorée.")
        return
    try:
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=bucket)
        ok(f"Bucket '{bucket}' accessible.")

        # Vérifie les dossiers
        for prefix in ["audio/", "transcriptions/"]:
            try:
                s3.head_object(Bucket=bucket, Key=prefix)
                ok(f"Dossier '{prefix}' présent.")
            except Exception:
                warn(f"Dossier '{prefix}' absent — lancez infra/setup_s3.py")
    except Exception as e:
        fail(f"Bucket S3 inaccessible : {e}")


# ─────────────────────────────────────────────
# 4. DYNAMODB
# ─────────────────────────────────────────────

def check_dynamodb() -> None:
    print("\n── DynamoDB ─────────────────────────────────────────")
    table_name = os.environ.get("DYNAMODB_TABLE", "FeedbackStore")
    region = os.environ.get("AWS_REGION", "us-east-1")
    try:
        dynamodb = boto3.client('dynamodb', region_name=region)
        response = dynamodb.describe_table(TableName=table_name)
        table = response['Table']
        ok(f"Table '{table_name}' trouvée — statut : {table['TableStatus']}")

        # Vérifie les GSI
        gsi_names = [g['IndexName'] for g in table.get('GlobalSecondaryIndexes', [])]
        for expected_gsi in ['meeting_id-index', 'score-index']:
            if expected_gsi in gsi_names:
                ok(f"GSI '{expected_gsi}' présent.")
            else:
                warn(f"GSI '{expected_gsi}' absent — lancez infra/setup_dynamodb.py")

        # Vérifie le TTL
        ttl = dynamodb.describe_time_to_live(TableName=table_name)
        ttl_status = ttl['TimeToLiveDescription']['TimeToLiveStatus']
        if ttl_status == 'ENABLED':
            ok("TTL activé.")
        else:
            warn(f"TTL non activé (statut: {ttl_status})")

    except dynamodb.exceptions.ResourceNotFoundException:
        fail(f"Table '{table_name}' introuvable — lancez infra/setup_dynamodb.py")
    except Exception as e:
        fail(f"Erreur DynamoDB : {e}")


# ─────────────────────────────────────────────
# 5. PINECONE
# ─────────────────────────────────────────────

def check_pinecone() -> None:
    print("\n── Pinecone ─────────────────────────────────────────")
    api_key    = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        fail("PINECONE_API_KEY ou PINECONE_INDEX_NAME manquants.")
        return

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        ok(f"Index '{index_name}' accessible.")
        ok(f"Dimension : {stats.get('dimension', '?')} (attendu : 1536)")
        ok(f"Vecteurs indexés : {stats.get('total_vector_count', 0)}")

        if stats.get('dimension') != 1536:
            fail("Dimension incorrecte ! L'index doit être configuré en 1536 dims.")

    except Exception as e:
        fail(f"Pinecone inaccessible : {e}")


# ─────────────────────────────────────────────
# 6. BEDROCK (modèles disponibles)
# ─────────────────────────────────────────────

def check_bedrock() -> None:
    print("\n── AWS Bedrock ──────────────────────────────────────")
    region = os.environ.get("AWS_REGION", "us-east-1")
    try:
        bedrock = boto3.client('bedrock', region_name=region)
        models = bedrock.list_foundation_models()['modelSummaries']
        model_ids = [m['modelId'] for m in models]

        for model in ['amazon.titan-embed-text-v1', 'anthropic.claude-haiku-4-5-20251001-v1:0']:
            # Vérifie si le modèle est disponible (match partiel)
            available = any(model in mid for mid in model_ids)
            if available:
                ok(f"Modèle disponible : {model}")
            else:
                warn(f"Modèle non listé (vérifiez l'accès Bedrock) : {model}")

    except Exception as e:
        fail(f"Bedrock inaccessible : {e}")


# ─────────────────────────────────────────────
# 7. AWS TRANSCRIBE
# ─────────────────────────────────────────────

def check_transcribe() -> None:
    print("\n── AWS Transcribe ───────────────────────────────────")
    region = os.environ.get("AWS_REGION", "us-east-1")
    try:
        transcribe = boto3.client('transcribe', region_name=region)
        # Liste les 5 derniers jobs pour vérifier l'accès
        transcribe.list_transcription_jobs(MaxResults=5)
        ok("AWS Transcribe accessible.")
    except Exception as e:
        fail(f"AWS Transcribe inaccessible : {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Vérification de l'environnement — Smart Meeting RAG")
    print("=" * 55)

    check_env_vars()
    check_aws_credentials()
    check_s3()
    check_dynamodb()
    check_pinecone()
    check_bedrock()
    check_transcribe()

    print("\n" + "=" * 55)
    print(f"  Résultat : {CHECKS_PASSED} ✅  {CHECKS_FAILED} ❌")
    print("=" * 55)

    if CHECKS_FAILED > 0:
        print("\n  Corrigez les erreurs ci-dessus avant de déployer.")
        sys.exit(1)
    else:
        print("\n  Environnement prêt. Vous pouvez déployer les Lambdas.")
        sys.exit(0)