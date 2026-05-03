import os
import json
import time
import requests
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://votre-api-gateway.amazonaws.com/prod")
S3_BUCKET    = os.environ.get("S3_BUCKET_NAME", "smart-meeting-rag-audio")

st.set_page_config(
    page_title="Smart Meeting RAG",
    page_icon="🎙️",
    layout="wide"
)

# ─────────────────────────────────────────────
# INITIALISATION SESSION STATE
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # historique du chat
if "last_query_id" not in st.session_state:
    st.session_state.last_query_id = None
if "last_chunks_used" not in st.session_state:
    st.session_state.last_chunks_used = []
if "last_meeting_id" not in st.session_state:
    st.session_state.last_meeting_id = None
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = False


# ─────────────────────────────────────────────
# HELPERS API
# ─────────────────────────────────────────────

def upload_to_s3(file_bytes: bytes, filename: str) -> bool:
    """Upload le fichier audio directement dans S3 via boto3."""
    import boto3
    s3 = boto3.client('s3')
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"audio/{filename}",
            Body=file_bytes
        )
        return True
    except Exception as e:
        st.error(f"Erreur S3 : {str(e)}")
        return False


def call_search_api(query: str, conversation_history: list) -> dict | None:
    """Appelle la Lambda search via API Gateway."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "conversation_history": conversation_history},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur API search : {str(e)}")
        return None


def call_feedback_api(query_id: str, score: int, query_text: str,
                      chunks_used: list, meeting_id: str) -> bool:
    """Envoie le feedback 👍/👎 à la Lambda feedback."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "query_id": query_id,
                "score": score,
                "query_text": query_text,
                "chunks_used": chunks_used,
                "meeting_id": meeting_id
            },
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Erreur API feedback : {str(e)}")
        return False


# ─────────────────────────────────────────────
# ONGLET 1 — UPLOAD AUDIO
# ─────────────────────────────────────────────

def render_upload_tab():
    st.header("🎙️ Indexer une nouvelle réunion")
    st.caption("Déposez un fichier audio (MP3, WAV, MP4). Il sera transcrit automatiquement et indexé.")

    uploaded_file = st.file_uploader(
        "Choisir un fichier audio",
        type=["mp3", "wav", "mp4", "m4a"],
        help="Max 500 MB. La transcription prend 1-5 min selon la durée."
    )

    if uploaded_file:
        st.audio(uploaded_file)
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🚀 Lancer l'indexation", type="primary", use_container_width=True):
                with st.spinner("Upload en cours..."):
                    success = upload_to_s3(uploaded_file.read(), uploaded_file.name)

                if success:
                    st.success(f"✅ **{uploaded_file.name}** uploadé dans S3.")
                    st.info(
                        "⏳ La Lambda d'ingestion va se déclencher automatiquement.\n\n"
                        "La transcription et l'indexation prennent **1 à 5 minutes** "
                        "selon la durée de l'audio. Vous pouvez interroger la réunion "
                        "depuis l'onglet **Chat** une fois l'indexation terminée."
                    )

        with col2:
            st.markdown("**Pipeline déclenché :**")
            st.code(
                f"S3 audio/{uploaded_file.name}\n"
                "  → Lambda ingestion (S3 trigger)\n"
                "  → AWS Transcribe (audio → texte)\n"
                "  → Chunking (300 mots, overlap 50)\n"
                "  → Bedrock Titan (embedding 1536 dims)\n"
                "  → Pinecone (upsert)",
                language="text"
            )

    st.divider()
    st.subheader("📂 Réunions disponibles pour la démo")
    demo_meetings = [
        {"nom": "reunion_budget_q1_2025.mp3",    "durée": "42 min", "chunks": 87,  "statut": "✅ Indexé"},
        {"nom": "kickoff_projet_alpha.mp3",       "durée": "28 min", "chunks": 58,  "statut": "✅ Indexé"},
        {"nom": "retrospective_sprint_12.mp3",    "durée": "35 min", "chunks": 71,  "statut": "✅ Indexé"},
        {"nom": "interview_technique_candidat.mp3","durée": "55 min", "chunks": 112, "statut": "✅ Indexé"},
    ]
    st.dataframe(demo_meetings, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ONGLET 2 — CHAT RAG
# ─────────────────────────────────────────────

def render_chat_tab():
    st.header("💬 Interroger vos réunions")
    st.caption("Posez une question en langage naturel sur le contenu de vos réunions indexées.")

    # Historique du chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources utilisées"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**Réunion :** `{src['meeting_id']}` — "
                            f"**Segment :** {src['chunk_index']} — "
                            f"**Score :** {src['score']}%"
                        )
                        st.caption(src.get("text_preview", ""))

    # Feedback sur la dernière réponse
    if (st.session_state.last_query_id
            and not st.session_state.feedback_sent
            and st.session_state.messages
            and st.session_state.messages[-1]["role"] == "assistant"):

        st.divider()
        st.caption("Cette réponse vous a-t-elle été utile ?")
        col1, col2, col3 = st.columns([1, 1, 8])

        with col1:
            if st.button("👍", key="thumbs_up", use_container_width=True):
                last_user_msg = next(
                    (m["content"] for m in reversed(st.session_state.messages)
                     if m["role"] == "user"), ""
                )
                if call_feedback_api(
                    st.session_state.last_query_id, 1,
                    last_user_msg,
                    st.session_state.last_chunks_used,
                    st.session_state.last_meeting_id or ""
                ):
                    st.session_state.feedback_sent = True
                    st.success("Merci pour votre retour !")
                    st.rerun()

        with col2:
            if st.button("👎", key="thumbs_down", use_container_width=True):
                last_user_msg = next(
                    (m["content"] for m in reversed(st.session_state.messages)
                     if m["role"] == "user"), ""
                )
                if call_feedback_api(
                    st.session_state.last_query_id, -1,
                    last_user_msg,
                    st.session_state.last_chunks_used,
                    st.session_state.last_meeting_id or ""
                ):
                    st.session_state.feedback_sent = True
                    st.warning("Noté. Ce feedback sera utilisé pour améliorer les résultats.")
                    st.rerun()

    # Input utilisateur
    if query := st.chat_input("Ex : Quand a-t-on décidé du budget Q2 ? Qui était présent ?"):
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.feedback_sent = False

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les réunions..."):
                # Convertit l'historique au format attendu par l'API
                history_for_api = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]
                result = call_search_api(query, history_for_api)

            if result:
                answer = result.get("answer", "Pas de réponse.")
                sources = result.get("sources", [])
                chunks_used = result.get("chunks_used", [])
                query_id = result.get("query_id")
                meeting_id = sources[0]["meeting_id"] if sources else None

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources utilisées"):
                        for src in sources:
                            st.markdown(
                                f"**Réunion :** `{src['meeting_id']}` — "
                                f"**Segment :** {src['chunk_index']} — "
                                f"**Score :** {src['score']}%"
                            )
                            st.caption(src.get("text_preview", ""))

                # Sauvegarde pour feedback
                st.session_state.last_query_id = query_id
                st.session_state.last_chunks_used = chunks_used
                st.session_state.last_meeting_id = meeting_id
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

    # Bouton reset
    if st.session_state.messages:
        if st.button("🗑️ Effacer la conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.last_query_id = None
            st.session_state.feedback_sent = False
            st.rerun()


# ─────────────────────────────────────────────
# ONGLET 3 — DASHBOARD MÉTRIQUES
# ─────────────────────────────────────────────

def render_dashboard_tab():
    st.header("📊 Observabilité du pipeline RAG")
    st.caption("Métriques en temps réel depuis CloudWatch — Namespace : `SmartMeetingRAG`")

    # Simulation de métriques pour la démo (à remplacer par appels CloudWatch réels)
    import random
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Requêtes totales",   "247",  "+12 aujourd'hui")
    col2.metric("Précision RAG",      "81%",  "+3% vs semaine dernière")
    col3.metric("Latence moyenne",    "1.4s", "-0.2s")
    col4.metric("Chunks réindexés",   "34",   "dernier run lundi")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Taux de feedback positif dans le temps")
        import pandas as pd
        import numpy as np

        # Données simulées — à remplacer par get_metric_statistics CloudWatch
        dates = pd.date_range(end=pd.Timestamp.now(), periods=14, freq='D')
        precision = [0.62, 0.65, 0.63, 0.68, 0.70, 0.69, 0.72,
                     0.74, 0.73, 0.76, 0.78, 0.80, 0.81, 0.81]
        df = pd.DataFrame({"date": dates, "précision": precision})
        df = df.set_index("date")
        st.line_chart(df, color="#1f77b4")
        st.caption("↑ Amélioration visible après chaque cycle de réindexation (tous les lundis)")

    with col_right:
        st.subheader("Feedback par jour (7 derniers jours)")
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        pos  = [18, 22, 15, 25, 20, 8, 5]
        neg  = [4, 3, 6, 2, 4, 1, 2]
        df2 = pd.DataFrame({"👍 Positif": pos, "👎 Négatif": neg}, index=days)
        st.bar_chart(df2, color=["#2ecc71", "#e74c3c"])

    st.divider()
    st.subheader("🔄 Historique des réindexations")
    reindex_history = [
        {"Date": "2025-04-28", "Réunions traitées": 3, "Chunks réindexés": 34,
         "Feedbacks traités": 12, "Précision avant": "72%", "Précision après": "81%"},
        {"Date": "2025-04-21", "Réunions traitées": 2, "Chunks réindexés": 21,
         "Feedbacks traités": 8,  "Précision avant": "65%", "Précision après": "72%"},
        {"Date": "2025-04-14", "Réunions traitées": 1, "Chunks réindexés": 9,
         "Feedbacks traités": 4,  "Précision avant": "62%", "Précision après": "65%"},
    ]
    st.dataframe(reindex_history, use_container_width=True, hide_index=True)
    st.caption("La réindexation tourne automatiquement chaque lundi à 2h UTC via EventBridge.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    st.title("🎙️ Smart Meeting RAG")
    st.markdown(
        "Pipeline RAG serverless sur transcriptions audio · "
        "AWS Transcribe · Bedrock · Pinecone · Feedback loop · CloudWatch"
    )

    tab1, tab2, tab3 = st.tabs([
        "📁 Indexer une réunion",
        "💬 Chat",
        "📊 Dashboard MLOps"
    ])

    with tab1:
        render_upload_tab()
    with tab2:
        render_chat_tab()
    with tab3:
        render_dashboard_tab()


if __name__ == "__main__":
    main()