import streamlit as st
import requests
import boto3

# 1. Configuration de la page
st.set_page_config(page_title="Smart ATS - RAG Complet", page_icon="💼", layout="wide")

# --- VARIABLES À CONFIGURER ---
# Ton URL API Gateway 
API_URL = "https://fx01ucrv74.execute-api.us-east-1.amazonaws.com/default/smart-ats-search" 
# Ton Bucket S3 (REMPLACE CETTE VALEUR !)
S3_BUCKET_NAME = "smart-ats-cv-dropzone-ryad-123" 
# ------------------------------

s3_client = boto3.client('s3')

st.title("💼 Smart ATS - Matching & Recherche Sémantique")
st.markdown("Propulsé par AWS Serverless (Bedrock, Titan, Claude 3.5) et Pinecone.")

# ==========================================
# 📂 BARRE LATÉRALE : UPLOAD DES CV VERS S3
# ==========================================
with st.sidebar:
    st.header("📥 Base de Candidats")
    st.markdown("Glissez les nouveaux CV ici. Ils seront automatiquement anonymisés, vectorisés et ajoutés à la base Pinecone.")
    
    uploaded_cvs = st.file_uploader("Déposez des CV (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Envoyer vers l'IA d'Ingestion", type="primary"):
        if not uploaded_cvs:
            st.warning("Veuillez sélectionner au moins un CV.")
        elif S3_BUCKET_NAME == "NOM_DE_TON_BUCKET_S3_ICI":
            st.error("N'oubliez pas de configurer la variable S3_BUCKET_NAME dans le code !")
        else:
            with st.spinner("Upload vers AWS S3 en cours..."):
                for cv in uploaded_cvs:
                    try:
                        # Envoi direct du fichier depuis l'ordinateur vers le Bucket S3
                        s3_client.upload_fileobj(cv, S3_BUCKET_NAME, cv.name)
                        st.success(f"✅ {cv.name} envoyé !")
                    except Exception as e:
                        st.error(f"Erreur S3 : Vérifiez vos identifiants AWS locaux. ({e})")
            st.info("💡 L'analyse IA se lance en arrière-plan. Les CV seront disponibles dans quelques secondes.")

# ==========================================
# FONCTION UTILITAIRE : AFFICHAGE DES CV
# ==========================================
def afficher_resultats(resultats):
    if not resultats:
        st.info("Aucun profil pertinent n'a été trouvé.")
        return
        
    st.success(f"🎯 {len(resultats)} profil(s) pertinent(s) trouvé(s) !")
    st.markdown("---")
    
    for i, res in enumerate(resultats):
        cv_name = res.get('fichier_cv', 'Inconnu')
        score = res.get('score_pertinence', 0)
        infos = res.get('donnees', {})
        
        with st.expander(f"🥇 Top {i+1} : {cv_name} (Pertinence : {score}%)", expanded=(i==0)):
            st.subheader(infos.get('titre_profil', 'Profil sans titre'))
            st.write(f"**Expérience estimée :** {infos.get('annees_experience_total', 0)} ans")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🛠️ Compétences Techniques :**")
                for comp in infos.get('competences_techniques', []):
                    st.markdown(f"- {comp}")
            with col2:
                st.markdown("**🤝 Soft Skills :**")
                for skill in infos.get('soft_skills', []):
                    st.markdown(f"- {skill}")
                    
            st.markdown("**💼 Expériences Clés :**")
            for exp in infos.get('experiences_cles', []):
                st.markdown(f"- {exp}")

# ==========================================
# 🎯 ZONE PRINCIPALE : LES DEUX ONGLETS
# ==========================================
tab1, tab2 = st.tabs(["⚡ Recherche Rapide", "📝 Matching sur Offre d'Emploi"])

# ONGLET 1 : RECHERCHE RAPIDE
with tab1:
    st.subheader("Recherche ponctuelle")
    query_rapide = st.text_input("Recherche par mots-clés ou phrase", placeholder="Ex: Étudiant Python et HTML...")
    
    if st.button("Lancer la recherche", key="btn_rapide"):
        if query_rapide:
            with st.spinner("Recherche en cours..."):
                try:
                    response = requests.post(API_URL, json={"query": query_rapide})
                    if response.status_code == 200:
                        afficher_resultats(response.json().get("resultats", []))
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")
        else:
            st.warning("Veuillez entrer une requête.")

# ONGLET 2 : MATCHING D'OFFRE D'EMPLOI
with tab2:
    st.subheader("Trouver les meilleurs profils pour un poste")
    offre_texte = st.text_area("Collez la description de l'offre d'emploi ici", height=250, 
                               placeholder="Titre du poste, missions, compétences requises (copier-coller de LinkedIn ou Welcome to the Jungle)...")
    
    if st.button("Lancer le Matching IA", key="btn_offre", type="primary"):
        if offre_texte:
            with st.spinner("🧠 Analyse de l'offre et matching vectoriel en cours..."):
                try:
                    # On envoie tout le texte de l'offre au moteur de recherche
                    response = requests.post(API_URL, json={"query": offre_texte})
                    if response.status_code == 200:
                        afficher_resultats(response.json().get("resultats", []))
                        # BIENTÔT : Ici on affichera le texte de justification de Claude !
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")
        else:
            st.warning("Veuillez coller le texte de l'offre d'emploi.")