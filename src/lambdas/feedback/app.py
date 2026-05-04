import os
import json
import boto3
from datetime import datetime, timezone

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "FeedbackStore")


# ─────────────────────────────────────────────
# 1. ÉCRITURE DYNAMODB
# ─────────────────────────────────────────────

def save_feedback(query_id: str, score: int, query_text: str,
                  chunks_used: list, meeting_id: str) -> None:
    """
    Enregistre le feedback utilisateur dans DynamoDB.
    score : 1 = positif (👍), -1 = négatif (👎)
    """
    table = dynamodb.Table(TABLE_NAME)
    timestamp = datetime.now(timezone.utc).isoformat()

    table.put_item(Item={
        "query_id": query_id,
        "timestamp": timestamp,
        "score": score,
        "query_text": query_text,
        "chunks_used": chunks_used,
        "meeting_id": meeting_id,
        "reindexed": False      # sera mis à True par la Lambda de réindexation
    })

    print(f"Feedback enregistré — query_id: {query_id}, score: {score}")


# ─────────────────────────────────────────────
# 2. MÉTRIQUES CLOUDWATCH
# ─────────────────────────────────────────────

def push_feedback_metrics(score: int) -> None:
    """Pousse les compteurs de feedback vers CloudWatch."""
    try:
        cloudwatch_client.put_metric_data(
            Namespace='SmartMeetingRAG',
            MetricData=[
                {
                    'MetricName': 'PositiveFeedback',
                    'Value': 1 if score == 1 else 0,
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'NegativeFeedback',
                    'Value': 1 if score == -1 else 0,
                    'Unit': 'Count'
                }
            ]
        )
    except Exception as e:
        print(f"Warning CloudWatch : {str(e)}")


# ─────────────────────────────────────────────
# 3. HANDLER LAMBDA
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Déclencheur : POST /feedback via API Gateway.
    Body attendu :
    {
        "query_id": "uuid",
        "score": 1 | -1,
        "query_text": "Question posée",
        "chunks_used": ["meeting1_chunk_3", ...],
        "meeting_id": "meeting_2025-05-03"
    }
    """
    try:
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event

        query_id   = body.get('query_id', '')
        score      = body.get('score')
        query_text = body.get('query_text', '')
        chunks_used = body.get('chunks_used', [])
        meeting_id = body.get('meeting_id', 'unknown')

        # Validation
        if not query_id:
            return {'statusCode': 400, 'body': json.dumps({"error": "'query_id' manquant."})}
        if score not in (1, -1):
            return {'statusCode': 400, 'body': json.dumps({"error": "'score' doit être 1 ou -1."})}

        save_feedback(query_id, score, query_text, chunks_used, meeting_id)
        push_feedback_metrics(score)

        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"message": "Feedback enregistré. Merci !"})
        }

    except Exception as e:
        print(f"Erreur feedback : {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }