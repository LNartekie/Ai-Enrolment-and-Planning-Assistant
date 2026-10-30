# Institutional Enrolment Forecasting and Planning Assistant

**An upload-based AI application for institutional enrolment forecasting, capacity planning, and document-grounded policy questions, supporting independent user sessions.**

Built with **Python, Streamlit, Ollama, Pydantic, NumPy, pandas, and statsmodels**, this application allows an institutional administrator to upload historical enrolment records, inspect trends, generate forecasts, evaluate projected capacity pressure, and ask questions about uploaded institutional policy documents. The application runs locally or in Docker.

> **Academic demonstration:** The sample institution, enrolment records, and policy documents used in the demonstration are fictional. Forecasts and generated explanations support human planning; they do not authorize admission, capacity expansion, or other institutional decisions.

## 1. Problem and objectives

Institutions need reliable ways to interpret historical student enrolment and compare future demand with their approved operating capacity. Manually combining spreadsheets, forecasts, and policy documents is time-consuming and can lead to misinterpretation—for example, confusing an *aspirational target* with an *approved capacity*.

The assistant is designed for **administrators and institutional planning staff from different institutions to use the same deployed application**, with each user working on their own institution's uploaded data in an independent session. Each forecasting analysis uses one institution's dataset; this is not an app-wide restriction to a single institution or user. Its goals are to:

- Validate uploaded institutional enrolment data and summarize historical trends.
- Generate a data-driven enrolment forecast with backtest metrics.
- Calculate forecast-versus-capacity metrics using deterministic Python code.
- Explain results through a local Ollama language model, including explicit tool use.
- Retrieve relevant passages from uploaded policy documents and display source references.
- Detect or surface invalid inputs, malformed AI outputs, and missing evidence where possible.

**Scope:** The application accepts an individual institution's data for each analysis, while the deployed service can be accessed by different users in separate sessions (including at the same time, subject to server capacity). It is **not** a shared multi-university admissions database. Formal concurrent-session isolation, authentication, access controls, and production-scale performance have **not yet been tested or implemented**, so it should not be described as a fully secured production multi-tenant platform.

## 2. Key features

| Capability | Implementation |
| --- | --- |
| Data upload and validation | CSV upload through Streamlit; validation of enrolment history |
| Trend analysis | Historical direction, annual changes, average and percentage change |
| Enrolment forecasting | Candidate time-series models including Naive, Linear Trend and Holt Linear Trend, with historical backtesting |
| Forecast interpretation | Local Ollama model; structured output with Pydantic validation |
| Conversational interface | Session-level conversation context and reset controls |
| Policy document ingestion | PDF, DOCX and TXT parsing, text cleaning, chunking and source metadata |
| Retrieval-augmented generation (RAG) | `nomic-embed-text` embeddings; in-memory similarity index; cited responses |
| Capacity planning | Python function to calculate capacity gap, utilisation and available places |
| LLM tool calling | Qwen requests an approved `analyse_capacity` tool; application validates arguments and executes Python |
| User experience | Streamlit progress indicators, validation messages, expandable references and readable metrics |
| Deployment | Dockerised Streamlit app; Ollama runs on the host in the tested setup |

The RAG workflow includes an experimental evidence-sufficiency evaluator that is **not used as a mandatory production gate**, because live tests showed it could incorrectly reject supported questions. The active structured RAG service checks response schemas and citation identifiers, but citation validity alone does not prove factual support.

## 3. Architecture overview

```text
Institutional administrator
          |
          v
     Streamlit UI (app/main.py)
          |
          +--- Enrolment CSV -> Data validation -> Trend analysis -> Forecasting
          |                                                |
          |                                                v
          |                                       Forecast + backtest metrics
          |                                                |
          |                                                v
          |                                       Capacity planning interface
          |                                                |
          |                                       Qwen requests approved tool
          |                                                |
          |                                       Python analyse_capacity()
          |                                                |
          |                                       Verified result + AI explanation
          |
          +--- Policy files -> Extraction -> Chunking -> Metadata
                                                       |
                                                       v
                                                Ollama embeddings
                                                       |
                                                       v
                                                Session vector index
                                                       |
                    User policy question -> Similarity retrieval
                                                       |
                                                       v
                                             Qwen structured RAG answer
                                                       |
                                                       v
                                            Pydantic/citation validation
                                                       |
                                                       v
                                               Answer and sources
```

**Model and service placement:** Streamlit and Python service modules run in the application process. The LLM and embedding model run in **Ollama**, contacted through its HTTP API. Application conversation state and document vectors are **session-scoped/in-memory**, not stored in PostgreSQL, MinIO or an external vector database.

## 4. Technology stack

- **Language:** Python (the provided Dockerfile uses Python 3.11)
- **Interface:** Streamlit
- **Data analysis:** pandas, NumPy, scikit-learn
- **Forecasting:** statsmodels, including Holt Linear Trend
- **AI model:** Ollama `qwen3:1.7b` for natural-language tasks and tool calls
- **Embedding model:** Ollama `nomic-embed-text`
- **Schema validation:** Pydantic
- **RAG index:** In-memory NumPy-backed session vector index
- **Document processing:** pypdf, python-docx; TXT support
- **Testing:** pytest
- **Deployment:** Docker and Docker Compose

### Why no PostgreSQL or MinIO?

The initial system supports **separate sessions for different users and institutions**, with each session analysing its own uploaded files. It does not currently maintain persistent user accounts, institution records, saved analyses or long-term document storage. PostgreSQL would be appropriate for institutional records, authenticated accounts, run history and audit logs; MinIO or another S3-compatible service would be appropriate for persistently storing uploaded documents, data files and generated reports. These are **future enhancements**, not silently present in the current build.

## 5. Project structure

Key files and directories in the implementation are shown below; your repository may also contain additional modules and experiment outputs.

```text
.
├── app/
│   ├── main.py                      # Streamlit entry point
│   ├── prompts/                     # Prompt variants
│   ├── schemas/
│   │   ├── forecast_response.py
│   │   └── rag_response.py
│   └── services/
│       ├── data_validation.py
│       ├── trend_analysis.py
│       ├── forecasting.py
│       ├── llm_service.py
│       ├── document_processing.py
│       ├── embedding_service.py
│       ├── retrieval_service.py
│       ├── rag_service.py
│       ├── structured_rag_service.py
│       ├── evidence_sufficiency.py  # Experimental component
│       ├── planning_tools.py
│       └── tool_calling_service.py
├── data/
│   └── demo/                        # Fictional example files, if included
├── evaluation/                      # Evaluation scripts and results
├── tests/                           # Unit/regression tests
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

## 6. Prerequisites

For **local execution**:

1. Python and pip (Python 3.11 is used in Docker; use a compatible local Python version).
2. [Ollama](https://ollama.com/) installed and running.
3. The following Ollama models downloaded:

```powershell
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
ollama list
```

For **Docker execution**, install Docker Desktop with Docker Compose. The tested configuration runs Ollama on the **Windows host**, while Streamlit runs inside the Docker container. Docker does not automatically download or start Ollama in this configuration.

## 7. Environment configuration

Create a local `.env` file if needed (do not commit secrets). For an application run directly on your computer:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

For the supplied Docker Compose configuration, these variables are set automatically:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:1.7b
PYTHONPATH=/app
```

The base URL differs because **`localhost` inside Docker refers to the container**, not your Windows host. Ensure Ollama is accessible to the container; if model requests fail, inspect Docker logs and verify host-to-container connectivity. Avoid exposing Ollama to untrusted networks.

## 8. Install and run locally

From the repository root, in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/main.py --server.port 8502
```

If `py -3.11` is unavailable but a compatible Python is installed, use that interpreter when creating the virtual environment.

Open **http://localhost:8502**. Ollama must be running separately before using AI features.

## 9. Run with Docker

The project was built and launched using Docker Engine **29.4.1** and Docker Compose **v5.1.3** on Windows. From the repository root:

```powershell
docker compose config
docker compose up -d --build
docker compose ps
```

Open **http://localhost:8503**. The provided Compose configuration maps host port **8503** to Streamlit port **8502** inside the container, so it can coexist with a local Streamlit instance on 8502.

Useful commands:

```powershell
docker compose logs --tail 80
docker compose down
```

Test whether the container can reach Ollama:

```powershell
docker compose exec enrolment-assistant python -c "import requests; r=requests.get('http://host.docker.internal:11434/api/tags', timeout=10); print(r.status_code); print([m['name'] for m in r.json().get('models', [])])"
```

A running container confirms startup, **not** that AI features work; perform the full workflow test below.

## 10. Demonstration workflow

1. Launch the application and upload a **historical enrolment CSV**. The basic supported demonstration schema is:

   ```csv
   year,enrolment
   2022,1540
   2023,1596
   2024,1668
   2025,1815
   ```

   This is an excerpt for illustration; forecasting requires adequate historical records. A separate 25-year synthetic dataset covering 2001–2025 was used in testing and is also available.

2. Review the historical trend and generate the **next-year forecast**. For the synthetic 2001–2025 demonstration, the observed result was **1,920 students for 2026** using **Holt Linear Trend**, with historical backtest **MAE 17.82**, **RMSE 25.14**, and **MAPE 1.45%**. These results are **dataset-specific** and not a general accuracy guarantee.
3. Upload the fictional **2020–2026 Institutional Enrolment and Capacity Planning Policy** in the policy-document section.
4. Ask: **“What happens when enrolment exceeds approved capacity?”** Verify that the response refers to the mandatory variance review, notifications and documented escalation, and inspect its source references.
5. Open **AI Capacity Planning Analysis**. The latest generated forecast now **automatically populates** the forecast field. Enter the **approved capacity** manually; for the fictional 2026 policy it is **1,800**. The 2026 **1,900 provisional target is not approved capacity**.
6. Click **Analyse Capacity**. The application invokes the allowlisted Python tool via Qwen and displays the authoritative computed metrics.

Expected figures for this demonstration:

| Metric | Value |
| --- | ---: |
| Forecast year | 2026 |
| Forecast enrolment | 1,920 |
| Approved operating capacity | 1,800 |
| Projected capacity gap (`forecast - capacity`) | **+120** |
| Projected capacity utilisation | **106.67%** |
| Remaining places | **0** |
| Planning status | **Over capacity** |

**Interpretation:** Enrolment is **projected** to exceed capacity by 120; it has not necessarily exceeded capacity in actual records. The policy requires a documented variance review if verified enrolment exceeds capacity or a reliable estimate shows material risk. A formal capacity increase requires separate institutional approval.

### Extended CSV

An additional synthetic CSV includes `capacity`, `new_intake`, `continuing_students`, `graduates`, `withdrawals`, `academic_staff`, `teaching_rooms` and `capacity_record_status`. Use the basic `year,enrolment` CSV first. Additional columns are optional demo context, **not a guarantee that the forecasting pipeline incorporates every field**. The pre-2020 capacity entries are intentionally blank because the fictional policy does not establish them.

## 11. RAG and source grounding

Policy files are extracted, cleaned and split into overlapping text chunks with source metadata such as filename, page and chunk ID. The embedding service uses Ollama `nomic-embed-text`; the session vector index retrieves relevant chunks for a user's query. Retrieved text and citation identifiers are sent to Qwen. The response is parsed and validated using Pydantic before source references are shown.

**Why RAG rather than putting all documents into the prompt?** Retrieval selects smaller, relevant portions of uploaded material, reducing unnecessary prompt context and allowing source attribution and session-scoped document updates.

**Known limitation:** An LLM may cite a real retrieved chunk that does **not** substantiate the requested claim. An unrelated airport-construction policy question was incorrectly labelled `answered` in one live evaluation. An experimental separate sufficiency evaluator rejected that query but also incorrectly rejected a supported capacity-exceedance question. Therefore it is not treated as a reliable mandatory gate. 

## 12. Approved tool execution and deterministic analysis

The capacity tool is implemented in `app/services/planning_tools.py`, with an Ollama interaction layer in `app/services/tool_calling_service.py`.

1. The application obtains a forecast and administrator-supplied approved capacity.
2. Pydantic validates the two nonnegative/positive integer inputs.
3. Qwen requests the allowlisted `analyse_capacity` function through Ollama's tool-calling interface.
4. Python validates the tool name and arguments **against the trusted application values** and executes the calculation.
5. Qwen receives the tool result and explains it in plain language. The displayed numerical metrics come from Python, not from the model's prose.

The model cannot call arbitrary Python functions or override validated application figures through tool arguments.

## 13. Evaluation and tests

Run all automated tests:

```powershell
python -m pytest tests -v
```

Run relevant component tests:

```powershell
python -m pytest tests/test_planning_tools.py tests/test_tool_calling_service.py -v
python -m pytest tests/test_rag_response.py tests/test_structured_rag_service.py -v
```

The development process included tests for data validation, forecasting, structured responses, document processing, retrieval, RAG, capacity calculations, tool calling and failure handling. **A count of 91 passing tests was explicitly confirmed in development**
Live checks were also performed for the Docker deployment, historical forecasting, policy Q&A and capacity planning. 

Observed failure cases and mitigations include:

- **False RAG abstention:** A required variance procedure initially ranked fourth, beyond top-3 retrieval. Increasing `top_k` to 5 and using structured responses helped the model find and answer the question in later tests.
- **Unsupported policy answer:** A question about airport construction produced a plausible but unsupported answer with citations. 
- **Malformed/contradictory model output:** Pydantic schemas, citation checks, one corrective retry and controlled UI errors mitigated some invalid responses.
- **Unavailable model or invalid input:** Service exceptions are caught and surfaced without using invented fallback calculations.

**Evaluation obligation:** The project requested at least **10 representative evaluation cases**. A separate acceptance checklist should record each case's actual result rather than mark unrun tests as passed.

## 14. Reliability, privacy and limitations

- The application is intended for use by **multiple independent users and institutions**. Streamlit maintains separate session state for separate connected browser sessions, so users can analyse different institutional datasets without a shared central upload database. This session separation should still be verified with two concurrent users before broader deployment; it is not a substitute for authenticated tenant isolation.
- Uploaded documents, conversation state and vectors are session-scoped rather than held in a persistent institutional database. Users may need to re-upload documents in a new session; do not treat session state as long-term storage.
- The present demo does **not** implement persistent PostgreSQL/MinIO storage, multi-user authentication, institution-level permissions, audited cross-session isolation, automated institution-to-institution benchmarking, or external approval workflows. Different people can use separate sessions; this does not imply production-grade user management.
- The **approved capacity** is entered by the administrator. A forecast or a provisional target must **never** be silently treated as an approved capacity.
- Forecast model accuracy depends on the uploaded institution's history. Historical backtesting cannot guarantee future performance; abrupt policy, economic or demographic changes can degrade forecasts.
- RAG source identifiers confirm that a referenced chunk was retrieved, **not** that every sentence in the generated answer is supported.
- The small local LLM may fail to follow tool-call or structured-answer instructions. The application rejects invalid tool calls rather than simulating success.
- Policy documents may include sensitive institutional information. Only upload documents that the user is authorized to process, and review local Ollama/Streamlit/Docker deployment settings before broader access.
- Demo data are fictional and are not evidence of real institutional decisions.

## 15. Future improvements

- Addition of a persistent database (e.g., PostgreSQL) for authenticated users, institutional histories, audit logs and saved runs.
- Addition of object storage (e.g., MinIO) for uploaded files and generated reports; enforce access control and retention policies.
- Improve source-grounding using retrieval reranking, support verification and larger model comparisons.
- Verify cross-session data isolation with simultaneous users and introduce authentication, institution-level permissions, and access controls for production deployment.
- Addition of forecast uncertainty ranges, scenario comparisons and capacity-approval workflow integration.
- Addition of automated end-to-end CI tests and deployment health checks.

---

**Project status:** Demonstration implementation; not a substitute for institutional approval or independently audited admissions planning.
