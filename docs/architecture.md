# Insurance Comparison Architecture

This document describes the current project architecture and the main runtime/data pipelines.

## 1. System Context

```mermaid
flowchart TB
    U[User]
    FE[Frontend]
    API[Backend API]

    U --> FE --> API

    subgraph SVC[Backend Core Flow]
        ORCH[Chat / Compare Orchestrator]
        ING[Ingestion Pipeline]
    end

    API --> ORCH
    API --> ING

    subgraph DATA[Data Layer]
        SQL[(SQLite<br/>plans / facts / sessions)]
        VDB[(Milvus<br/>search index)]
        OUT[(Output Artifacts<br/>document & facts files)]
    end

    subgraph SRC[Input Source]
        PDF[(Policy Documents)]
    end

    subgraph EXT[Model Services]
        LLM[(LLM & Embeddings)]
    end

    ING --> PDF
    ING --> SQL
    ING --> VDB
    ING --> OUT
    ING --> LLM

    ORCH --> SQL
    ORCH --> VDB
    ORCH --> LLM
```

System context responsibilities:

- Ingestion pipeline: read policy docs, extract structured facts, write data stores, and refresh retrieval index.
- Chat/compare orchestrator: process user intent, keep session state, run compare, and trigger plan discovery when needed.
- Data layer: SQLite is the source of truth for table/state; Milvus is retrieval acceleration for semantic discovery/evidence.

## 2. Chat + Auto Discover Runtime Flow

```mermaid
sequenceDiagram
    participant User
    participant FE as Frontend
    participant API as Backend API
    participant Orchestrator as Chat/Compare Orchestrator
    participant LLM as LLM
    participant Retriever as Retrieval
    participant Compare as Compare Engine
    participant SQL as SQLite
    participant Milvus as Milvus

    User->>FE: Ask question
    FE->>API: POST /api/chat/message/stream
    API->>Orchestrator: Process message + session state
    Orchestrator->>LLM: Parse intent and focus target

    alt Condition/Surgery question
        Orchestrator->>Retriever: Check selected plans evidence
        Retriever->>Milvus: Retrieve relevant chunks
        Orchestrator->>LLM: Judge coverage based on evidence
        alt Selected plans do not cover
            Orchestrator->>Retriever: Discover better-matched plans
            Retriever->>Milvus: Retrieve candidate plans
        end
    end

    Orchestrator->>Compare: Build/refresh compare result
    Compare->>SQL: Read structured facts
    Orchestrator-->>API: reply + state + compare + turns
    API-->>FE: SSE token stream + done payload
    FE-->>User: update chat and compare table
```

## 3. Ingestion Pipeline

```mermaid
flowchart TD
    A[Trigger ingestion] --> B[Load policy documents]
    B --> C[Parse content and split into chunks]
    C --> D[Extract facts and dimensions]
    D --> E[Normalize and clean values]
    E --> F[Upsert structured records to SQLite]
    E --> G[Upsert chunks/facts to Milvus collections]
    F --> H[Rewrite output artifacts for review]
    G --> I[Collections ready for semantic retrieval]
    H --> J[Data ready for compare and chat discovery]
    I --> J
```

## 4. Core SQLite Data Model

```mermaid
erDiagram
    PLANS ||--o{ POLICY_CHUNKS : has
    PLANS ||--o{ POLICY_FACTS : has
    POLICY_CHUNKS ||--o{ POLICY_FACTS : sources
    CHAT_SESSIONS ||--o{ CHAT_TURNS : has
    CHAT_SESSIONS ||--|| SESSION_STATE : owns

    PLANS {
      string plan_id PK
      string name
      string source_file
    }

    POLICY_CHUNKS {
      string chunk_id PK
      string plan_id FK
      int page_start
      string section_path
      text text
    }

    POLICY_FACTS {
      string fact_id PK
      string plan_id FK
      string dimension_key
      text value_text
      string normalized_value
      string source_chunk_id FK
      int source_page
      float confidence
    }

    CHAT_SESSIONS {
      string session_id PK
      string user_id
      datetime created_at
    }

    CHAT_TURNS {
      string turn_id PK
      string session_id FK
      string role
      text content
    }

    SESSION_STATE {
      string session_id PK
      json selected_plans
      json dimensions
      json filters
      json last_table_snapshot
    }
```

## 5. Frontend Composition (Current)

- `App.jsx`: page shell and global state (`selectedPlanIds`, `selectedDimensions`, `compareData`, `sessionId`, `chatTurns`).
- `PlanDimensionPanel.jsx`: choose plans and dimensions.
- `CompareTable.jsx`: render comparison rows and differences.
- `ChatPanel.jsx`: send message, render streaming/non-stream responses.
- `api.js`: REST + SSE integration with backend.
