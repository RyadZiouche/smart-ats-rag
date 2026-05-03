# 💼 Smart ATS — Moteur de Recherche Sémantique de CV (RAG 100% Serverless)

![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)

## 📌 Contexte

Le tri de CV est souvent l'une des tâches les plus chronophages pour les équipes RH. Les moteurs de recherche classiques par mots-clés montrent rapidement leurs limites : ils ignorent le contexte, la synonymie et l'expérience globale d'un candidat.

**La solution :** un Applicant Tracking System (ATS) intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation), capable de comprendre le langage naturel et de faire correspondre sémantiquement des CV avec des offres d'emploi ou des requêtes spécifiques.

---

## 🚀 Fonctionnalités

- **Upload et ingestion automatisés :** dépôt de CV (PDF) directement depuis l'interface vers Amazon S3.
- **Anonymisation RGPD et extraction :** utilisation de **Claude Haiku (AWS Bedrock)** pour extraire et structurer les compétences tout en supprimant les données personnelles.
- **Vectorisation sémantique :** transformation des profils en vecteurs mathématiques (1 536 dimensions) via **Amazon Titan**.
- **Recherche en langage naturel :** interrogation de la base de données vectorielle **Pinecone** avec des phrases simples (ex. : *« Je cherche un développeur Python avec 2 ans d'expérience »*).
- **Matching d'offres d'emploi :** soumission d'une fiche de poste complète pour trouver les profils les plus pertinents.
- **Interface utilisateur :** front-end interactif développé avec **Streamlit**.

---

## 🏗️ Architecture Cloud (100% Serverless)

L'application repose sur une infrastructure AWS native, événementielle et hautement scalable.

graph TD
    %% Définition des couleurs
    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:white;
    classDef front fill:#FE4B4B,stroke:#232F3E,stroke-width:2px,color:white;
    classDef db fill:#000000,stroke:#232F3E,stroke-width:2px,color:white;

    User([👤 Recruteur]) -->|Interagit avec| UI(🖥️ Interface Streamlit):::front

    subgraph AWS [Cloud AWS - 100% Serverless]
        UI -->|1. Upload CV| S3[(Amazon S3\nStockage PDF)]:::aws
        S3 -->|2. Déclenche| L1(⚡ Lambda Ingestion):::aws
        L1 -->|3. Parse & Anonymise| B1{🧠 Bedrock\nClaude 4.5}:::aws
        L1 -->|4. Vectorise| B2{🧠 Bedrock\nTitan Embedding}:::aws
        
        UI -->|A. Requête de recherche| API(🌐 API Gateway):::aws
        API -->|B. Route vers| L2(⚡ Lambda Search):::aws
        L2 -->|C. Vectorise la requête| B2
    end

    subgraph External [Base Vectorielle]
        L1 -->|5. Stocke les Vecteurs| PC[(🌲 Pinecone)]:::db
        L2 -->|D. Recherche de Similarité| PC
        PC -->|E. Renvoie Top 3 CV| L2
    end
    
    L2 -->|F. Affiche les résultats| UI

**Pipeline de traitement (backend) :**

1. **Amazon S3 :** stockage brut des PDF entrants.
2. **AWS Lambda — Ingestion :** déclenchée automatiquement par S3 ; extrait le texte, appelle Bedrock et pousse les données vers Pinecone.
3. **AWS Bedrock (Claude Haiku) :** LLM utilisé pour le parsing JSON et l'anonymisation stricte.
4. **AWS Bedrock (Amazon Titan) :** modèle d'embedding pour la création des vecteurs.
5. **Pinecone :** base de données vectorielle gérant l'indexation et la recherche par similarité cosinus.
6. **Amazon API Gateway :** point d'entrée HTTP sécurisé.
7. **AWS Lambda — Search :** reçoit les requêtes via l'API, les vectorise, interroge Pinecone et renvoie les résultats formatés.

---

## 📂 Structure du projet

```text
smart-ats-rag/
│
├── data/
│   └── raw_cvs/                # CV de test en local
│
├── src/
│   ├── frontend/
│   │   └── app.py              # Interface Streamlit
│   │
│   └── lambdas/
│       ├── ingestion/
│       │   └── app.py          # Lambda d'ingestion (S3 → Bedrock → Pinecone)
│       │
│       ├── chat/
│       │   └── app.py          # Lambda de recherche (API Gateway → Bedrock → Pinecone)
│       │
│       └── feedback/           # Module futur (Human-in-the-loop)
│
└── tests/
    └── test_ingestion.py       # Tests locaux pour le parsing et l'embedding
```

---

## 🛠️ Installation et lancement en local

### Prérequis

- Un compte AWS avec accès à Bedrock et S3
- Un compte Pinecone avec un index configuré en 1 536 dimensions
- Python 3.11+

### Étapes

1. Clonez le dépôt :

```bash
   git clone https://github.com/votre-nom/smart-ats-rag.git
   cd smart-ats-rag
```

2. Installez les dépendances :

```bash
   pip install streamlit boto3 requests pypdf pinecone-client python-dotenv
```

3. Lancez l'interface Streamlit :

```bash
   cd src/frontend
   streamlit run app.py
```