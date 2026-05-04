# 🎙️ Smart Meeting RAG — Serverless RAG Pipeline with Observability

![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Terraform](https://img.shields.io/badge/Terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

> Pipeline RAG serverless de bout en bout sur transcriptions audio de réunions — avec feedback loop, réindexation automatique et observabilité CloudWatch.

---

## 📌 Problème & Solution

Les réunions d'entreprise génèrent une quantité massive d'informations non structurées, impossibles à retrouver après coup. Les outils de recherche classiques par mots-clés ignorent le contexte et la sémantique.

**La solution :** un pipeline MLOps serverless qui transcrit automatiquement les réunions audio, les indexe sémantiquement, et permet de les interroger en langage naturel — avec un système de feedback loop qui améliore la précision du RAG dans le temps.

---

## 🏗️ Architecture

```
Audio MP3/WAV
     │
     ▼
Amazon S3  ──(trigger)──▶  Lambda Ingestion
                                   │
                          AWS Transcribe
                          (audio → texte)
                                   │
                            Chunking
                         (300 mots, overlap 50)
                                   │
                       Bedrock Titan Embed
                           (1536 dims)
                                   │
                              Pinecone
                           (Vector Store)
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
     Lambda Search                             Lambda Reindexation
   (API Gateway)                              (EventBridge — lundi 2h)
              │                                         │
    Bedrock Claude Haiku                        DynamoDB Scan
    (RAG generation)                         (feedbacks négatifs)
              │                                         │
         Streamlit UI                         Re-chunking fin
       (Chat + 👍/👎)                       (150 mots) + Re-embed
              │                                         │
     Lambda Feedback                            Pinecone Upsert
              │
          DynamoDB
       (FeedbackStore)
              │
         CloudWatch
    (métriques + alarmes)
```

---

## 🚀 Fonctionnalités

**Pipeline d'ingestion event-driven**
Upload d'un fichier audio → S3 déclenche automatiquement la Lambda d'ingestion → AWS Transcribe convertit l'audio en texte → découpe en chunks de 300 mots avec overlap → vectorisation Bedrock Titan (1536 dims) → indexation Pinecone.

**Chat RAG conversationnel**
Interrogation en langage naturel avec historique de conversation. Claude Haiku génère des réponses sourcées en citant la réunion et le segment exact. Top-k=5 chunks récupérés par similarité cosinus.

**Feedback loop human-in-the-loop**
Chaque réponse est évaluable via 👍/👎. Les feedbacks sont stockés dans DynamoDB avec les chunk IDs utilisés, le score et la question posée.

**Réindexation automatique**
Chaque lundi à 2h UTC, EventBridge déclenche la Lambda de réindexation qui relit les feedbacks négatifs, identifie les chunks mal retrouvés et les réindexe avec une stratégie de chunking plus fine (150 mots). La courbe de précision RAG s'améliore dans le temps.

**Observabilité complète**
Métriques custom CloudWatch : latence de recherche, score de similarité top-1, compteurs de feedback positif/négatif, chunks réindexés. Alarme SNS déclenchée si le taux de feedbacks négatifs dépasse le seuil en 24h.

---

## 🛠️ Stack Technique

| Composant | Service | Rôle |
|---|---|---|
| Stockage audio | Amazon S3 | Bucket avec versioning + lifecycle rules |
| Transcription | AWS Transcribe | Audio → texte structuré |
| LLM Génération | Bedrock Claude Haiku | Réponses RAG conversationnelles |
| Embedding | Bedrock Amazon Titan | Vectorisation 1536 dims |
| Vector DB | Pinecone Serverless | Indexation + recherche cosinus |
| Feedback store | DynamoDB | Feedbacks + 2 GSI + TTL 90j |
| Orchestration | AWS Lambda (x4) | Ingestion, Search, Feedback, Reindexation |
| API | API Gateway HTTP | Routes POST /search et /feedback |
| Scheduler | EventBridge | Cron hebdomadaire réindexation |
| Monitoring | CloudWatch | Métriques custom + alarmes SNS |
| IaC | Terraform | 31 ressources AWS versionnées |
| Frontend | Streamlit | UI 3 onglets (Upload, Chat, Dashboard) |

---

## 📂 Structure du Projet

```
smart-meeting-rag/
│
├── infra/
│   ├── main.tf                   # 31 ressources AWS (S3, Lambda, DynamoDB, API GW...)
│   ├── variables.tf              # Déclaration des variables
│   ├── outputs.tf                # API URL, ARNs exportés après apply
│   ├── terraform.tfvars.example  # Template de configuration
│   └── check_env.py              # Sanity check CI/CD (credentials, services)
│
├── src/
│   ├── frontend/
│   │   └── app.py                # Streamlit : Upload, Chat, Dashboard MLOps
│   │
│   └── lambdas/
│       ├── ingestion/
│       │   └── app.py            # S3 trigger → Transcribe → chunk → Titan → Pinecone
│       ├── search/
│       │   └── app.py            # query → embed → Pinecone → Claude Haiku → réponse
│       ├── feedback/
│       │   └── app.py            # 👍/👎 → DynamoDB + métriques CloudWatch
│       └── reindexation/
│           └── app.py            # EventBridge → feedbacks négatifs → réindexation
│
├── tests/
│   └── test_ingestion.py
│
├── .env.example
└── README.md
```

---

## ⚙️ Installation & Déploiement

### Prérequis

- AWS CLI configuré (`aws configure`)
- Terraform >= 1.5
- Python 3.11+
- Compte Pinecone (index 1536 dims, cosine, serverless, us-east-1)

### 1. Cloner le repo

```bash
git clone https://github.com/RyadZiouche/smart-meeting-rag.git
cd smart-meeting-rag
```

### 2. Configurer les variables

```bash
cp .env.example .env
# Remplir .env avec vos valeurs

cp infra/terraform.tfvars.example infra/terraform.tfvars
# Remplir terraform.tfvars avec vos valeurs
```

### 3. Déployer l'infrastructure

```bash
cd infra
terraform init
terraform plan    # visualiser les 31 ressources
terraform apply   # déployer (~2 min)
```

L'output `api_base_url` est à copier dans votre `.env`.

### 4. Vérifier l'environnement

```bash
python infra/check_env.py
# Doit afficher 17/17 ✅
```

### 5. Lancer le frontend

```bash
cd src/frontend
streamlit run app.py
```

---

## 🔄 Le Cycle MLOps

Ce projet implémente le cycle complet **Serve → Monitor → Retrain** appliqué à un pipeline RAG :

```
1. SERVE       Upload audio → transcription → indexation → recherche
2. COLLECT     Feedback utilisateur 👍/👎 stocké dans DynamoDB
3. MONITOR     CloudWatch : précision RAG, latence, taux de feedback négatif
4. RETRAIN     EventBridge weekly → réindexation des chunks mal retrouvés
               → amélioration mesurable de la précision dans le temps
```

C'est exactement ce qu'on attend d'un pipeline MLOps en production — pas juste "brancher une API", mais mesurer, monitorer et améliorer en continu.

---

## 📊 Métriques CloudWatch (Namespace : SmartMeetingRAG)

| Métrique | Description |
|---|---|
| `SearchLatencyMs` | Latence end-to-end de chaque requête |
| `TopChunkSimilarity` | Score cosinus du chunk le plus proche |
| `SearchCount` | Nombre total de requêtes |
| `PositiveFeedback` | Compteur 👍 |
| `NegativeFeedback` | Compteur 👎 |
| `ReindexedChunks` | Chunks réindexés par cycle |
| `ProcessedNegativeFeedbacks` | Feedbacks traités par cycle |

---

## 🔐 Sécurité

- Bucket S3 avec accès public bloqué
- Variables sensibles dans `terraform.tfvars` (jamais commité)
- IAM role avec least-privilege policy (actions spécifiques par service)
- TTL DynamoDB : suppression automatique des feedbacks après 90 jours
- Clé Pinecone injectée comme variable d'environnement Lambda (sensitive = true dans Terraform)

---

## 💡 Pistes d'Amélioration

- Remote backend Terraform (S3 + DynamoDB state locking) pour le travail en équipe
- CI/CD GitHub Actions : `terraform plan` sur PR, `terraform apply` sur merge main
- A/B test Titan v1 vs Titan v2 pour comparer la qualité des embeddings
- Support multilingue AWS Transcribe (fr-FR / en-US détection automatique)
- Lambda layers pour les dépendances Python communes (boto3, pinecone)

---

## 👤 Auteur

**Ryad Ziouche**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/ryadziouche)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/RyadZiouche)
