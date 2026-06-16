# Master Project Reverse Engineering Documentation

## 1. Project Overview

### Project Name
KCET College Finder and College RAG Chatbot

### Main Goal
The project helps a student explore KCET cutoff PDFs and supporting documents in a structured, searchable, and conversational way. The user can upload cutoff files, filter colleges by rank, round, category, and branch, compare colleges side by side, browse branches, and ask natural language questions from uploaded documents.

### Business Problem Solved
KCET cutoff data is usually published in PDF format. That means students must manually search many pages, compare rounds, interpret category codes, and mentally filter branches. This project reduces that friction by:
- extracting structured rows from PDFs
- converting PDF tables into searchable data
- giving ranked shortlist outputs
- providing a RAG assistant for document-based questions

### Target Users
- KCET students
- parents helping with admissions
- counselors and mentors
- college admission advisors
- anyone needing quick access to cutoff trends

### Real-World Use Cases
- “I scored 18,000 rank. What computer science colleges can I get in round 3?”
- “Show me all colleges for GM category only.”
- “Compare these two colleges side by side.”
- “What does this cutoff PDF say about information science branches?”
- “Which branches are available across the uploaded documents?”

### Expected Inputs
- PDF cutoff files
- CSV or Excel cutoff files
- optional supporting PDFs for RAG chat
- user rank
- round
- category
- branch filter

### Expected Outputs
- eligible college table
- side-by-side comparison cards
- branch browsing summary
- natural language answers with sources

---

## 2. Complete Architecture

### High-Level Architecture

User
↓
Streamlit Frontend
↓
Parsing + Filtering + RAG Orchestration
↓
Pandas / PyPDF2 / Chroma / LangChain
↓
Structured Results + LLM Answers

### Detailed Flow

1. The user uploads cutoff documents
2. The app detects file types
3. PDFs are parsed into text
4. Table rows are extracted from cutoff pages
5. The structured dataframe is built
6. The user chooses a mode
7. The app filters or retrieves accordingly
8. Results are displayed in the UI

### Connections Between Layers

- Streamlit handles the user interface
- Pandas handles the shortlist logic
- PyPDF2 handles raw PDF extraction
- Chroma stores embeddings for document Q&A
- LangChain connects retrieval and generation
- FastAPI exposes the chatbot as an API route

---

## 3. Folder Structure

### Top-Level Files

- `main.py`
- `vectorize_documents.py`
- `README.md`
- `requirements.txt`
- `LICENSE`
- `PROJECT_DOCUMENTATION.md`
- `MASTER_PROJECT_DOCUMENTATION.md`

### Important Folders

- `data/`
- `vector_db_dir/`
- `images/`
- `venv/`

### Meaning of Each Folder

#### `data/`
Contains source documents and templates used by the app.

#### `vector_db_dir/`
Contains the persistent vector database used for retrieval.

#### `images/`
Contains screenshots and unrelated image assets used by the repo.

#### `venv/`
Contains the local Python environment.

---

## 4. Design Pattern Used

The project uses a practical hybrid architecture:
- **UI layer**: Streamlit
- **API layer**: FastAPI
- **Parsing layer**: PDF/table extraction
- **Business logic layer**: rank and branch filtering
- **Retrieval layer**: Chroma + LangChain

This is not a strict MVC app, but the responsibilities are still separated in a layered manner.

### Why this pattern
- easy to build
- easy to demonstrate
- easy to extend
- good for both structured search and RAG chat

### Alternatives
- pure React frontend + separate backend
- Django full-stack app
- Flask app with Jinja templates

### Why those were not chosen
- Streamlit is faster for a student-facing prototype
- the project is document-heavy and interactive, not form-heavy

---

## 5. File-by-File Analysis

## 5.1 `main.py`

### Purpose
This is the central application file. It contains almost all the important logic:
- app setup
- file upload handling
- PDF parsing
- category normalization
- branch normalization
- rank filtering
- compare mode
- browse mode
- chat mode
- FastAPI endpoint

### Why it exists
The repo needs one entry point that can run the UI and the chatbot logic. This file coordinates the entire experience.

### Imports

- `fastapi`, `HTTPException`
- `BaseModel`
- `uvicorn`
- `os`, `json`, `re`
- `BytesIO`
- `streamlit`
- `pandas`
- `HuggingFaceEmbeddings`
- `Chroma`
- `ConversationBufferMemory`
- `ConversationalRetrievalChain`
- `PromptTemplate`
- `FPDF`
- `datetime`
- `PdfReader`
- optional `ChatGroq`
- optional `ChatOllama`

### Why those imports are used
- FastAPI handles API exposure
- Streamlit handles the UI
- Pandas handles shortlist filtering
- PyPDF2 reads PDFs
- Chroma retrieves document context
- LangChain handles LLM orchestration
- FPDF supports PDF generation features

---

### Key Global Configuration

#### `working_dir`
Current project directory.

#### `config_path`
Location of `config.json`.

#### `GROQ_API_KEY`
Read from config if available.

#### Why config is read at startup
This allows the app to work in both:
- API-key mode
- local Ollama mode

---

### FastAPI Section

#### `app = FastAPI()`
Creates the backend application.

#### `MessageRequest`
Pydantic model for incoming chatbot requests.

#### `/chat`
POST endpoint that:
- takes a message
- loads the vector store
- builds the conversational chain
- checks for sensitive topics
- returns an answer

### Why this endpoint exists
It makes the chatbot callable by another app or client.

### Alternatives
- pure Streamlit only
- Flask backend
- no API at all

### Why FastAPI is helpful
- typed request validation
- easy REST support
- modern architecture

---

### Prompt Design

`DEFAULT_SYSTEM_PROMPT` and `DEFAULT_NEGATIVE_PROMPT` define the chatbot behavior.

#### Purpose
- constrain answers to the college domain
- reduce hallucination
- keep answers concise and helpful

#### Why prompts matter
Prompts are the main control surface for model behavior when using RAG.

---

### Retrieval Helpers

#### `setup_vectorstore()`
Loads the persistent Chroma store from `vector_db_dir`.

##### Why it exists
It provides the base knowledge store for the chatbot.

##### Alternative
Could have used FAISS or Pinecone.

##### Why Chroma
- local persistence
- simple setup
- good enough for student projects

#### `get_base_vectorstore()`
Cached wrapper around `setup_vectorstore`.

##### Why cache it
Vector stores are expensive to recreate every rerun in Streamlit.

---

### Cutoff Loading Helpers

#### `load_cutoff_data()`
Reads fallback cutoff data from `data/kcet_cutoffs.csv` or `data/kcet_cutoffs_template.csv`.

##### Why it exists
Provides a default dataset if no upload is provided.

#### `load_cutoff_file(uploaded_file)`
Reads structured CSV/Excel uploads.

##### Expected columns
- `college_name`
- `branch`
- `round`
- `category`
- `cutoff_rank`
- `year`

##### Why strict column checks are useful
They prevent the app from trying to filter malformed data.

#### `load_uploaded_pdfs(uploaded_files)`
Extracts text from uploaded PDFs.

##### Why this exists
It supports both:
- cutoff parsing
- chat retrieval

---

### Round Detection

#### `infer_round_number(round_text, source_name)`
Infers round from title or filename.

##### Why needed
KEA-style PDFs do not always store round information in a clean database field.

##### Example
- filename contains `1109` -> round 3
- filename contains `3008` -> round 2
- filename contains `R1` -> round 1

##### Alternative
Hardcode round manually per file.

##### Why not chosen
Manual annotation is slow and error-prone.

---

### Category Logic

#### `normalize_category_query(category)`
Normalizes user category input.

##### Important behavior
- `GM` stays only `GM`
- `SC` expands to `SCG`, `SCK`, `SCR`
- `ST` expands to `STG`, `STK`, `STR`
- `2A`, `2B`, `3A`, `3B` expand to family variants

##### Why exact GM matters
The user explicitly requested no GM-family widening for `GM`.

##### Why family expansion is useful for some categories
Some KCET categories are naturally variant-based.

---

### Branch Logic

#### `clean_branch_name(branch)`
Removes bracketed text and excess whitespace.

#### `normalize_branch_query(branch)`
Creates a normalized search string.

#### `branch_family_name(branch)`
Maps messy branch titles into cleaner display names.

##### Why branch cleanup matters
Cutoff PDFs often contain long and inconsistent branch strings.

##### Example
Input:
- `Computer Science and Engineering (Artificial Intelligence and Machine Learning)`

Output:
- `Computer Science and Engineering`

##### Alternative
Show raw strings.

##### Why not chosen
Raw strings are hard for students to read.

---

### PDF Parsing

#### `_extract_fixed_columns(compact_values, expected_count)`
Attempts to split compact numeric data into fixed columns.

#### `extract_cutoff_rows_from_page(page_text, source_name)`
The core PDF parser.

##### What it does
- identifies college sections
- reads college code and college name
- reads round text
- reads category columns
- collects course rows
- maps cutoff values to category codes

##### Why it exists
KEA PDFs are table-like, but not actually structured like a normal dataframe.

##### Potential issues
- merged digits
- wrapped branch names
- scanned pages
- multiple colleges per page

##### Why the parser is written defensively
Because PDF extraction is messy and inconsistent.

---

### Combined Upload Loader

#### `load_uploaded_cutoff_data(uploaded_files)`
Unifies CSV, Excel, and PDF handling into one dataframe.

##### Why it exists
The app should accept whatever file type the student has.

##### Output
DataFrame of parsed cutoff rows.

---

### Vector Store for Uploaded PDFs

#### `build_uploaded_pdf_vectorstore(uploaded_files)`
Creates a temporary vector database from uploaded PDFs.

##### Why it exists
Supports chat questions about uploaded documents in the same session.

##### Alternative
Only use the prebuilt base vector DB.

##### Why not chosen
User-uploaded documents should be searchable immediately.

---

### Rank Window

#### `round_window(round_no)`
Defines the upper rank tolerance by round.

Current behavior:
- round 1 -> 3000
- round 2 -> 2800
- round 3 -> 2800

##### Why this exists
It reflects admission trend exploration instead of rigid exact cutoff matching.

##### Alternative
Symmetric plus/minus matching.

##### Why not chosen
Students usually care about nearby higher cutoffs more than symmetric bounds.

---

### Matching Logic

#### `find_kcet_matches(df, rank, round_no=None, category=None, branch=None, tolerance=None)`
Filters the dataframe based on user criteria.

##### Execution flow
1. create rank window
2. filter rows by cutoff range
3. filter by round
4. filter by category
5. filter by branch
6. sort results

##### Output
Matching dataframe rows.

##### Why this function is central
It directly decides what the student sees in shortlist mode.

---

### Shortlist Output

#### `branch_college_list(df)`
Formats matches into a display table.

##### Why it exists
The raw dataframe can be messy; this converts it into a user-facing shortlist.

##### Output columns
- `college_name`
- `branch`
- `round`
- `category`
- `cutoff`
- `year`

##### Why duplicates are removed
It prevents repeated identical rows from cluttering the UI.

---

### Summary Helpers

#### `build_data_summary(df)`
Creates summary metrics.

##### Why it exists
It tells the user if any useful data loaded.

#### `sorted_unique(values)`
Returns a clean unique list for dropdowns and selectors.

---

### RAG Chat

#### `has_relevant_context(vectorstore, question, threshold=0.35)`
Checks whether the vector store has a strong enough match.

##### Why
Prevents the LLM from answering when the context is weak.

#### `setup_llm()`
Selects Groq or Ollama.

##### Why this matters
It makes the app run in both cloud and local modes.

#### `chat_chain(vectorstore, system_prompt, negative_prompt)`
Builds the conversational retrieval chain.

#### `answer_question(vectorstores, conversational_chain, question)`
Runs the retrieval check and returns the final answer.

##### Why this wrapper exists
It adds a relevance gate before generation.

---

### Streamlit UI

The UI in `main.py` does the following:
- sets the page config
- shows the upload section
- loads data
- shows summary metrics
- lets the user choose a mode
- shows the shortlist/comparison/browse/chat output

##### Why Streamlit is appropriate
This project is highly interactive and data-driven.

---

### Search Modes in `main.py`

#### Rank shortlist
The main operational mode.

#### Compare colleges
Side-by-side comparison UI.

#### Browse branches
Aggregated branch summary.

#### Ask questions
RAG chat mode.

---

## 5.2 `vectorize_documents.py`

### Purpose
Builds the persistent vector database from PDFs in the `data` folder.

### Why it exists
Separates document ingestion from runtime query handling.

### Main steps
1. load PDFs
2. extract text
3. split into chunks
4. create embeddings
5. persist to Chroma

### Why this separation is good
- keeps startup clean
- allows rebuilding when data changes
- avoids mixing ingestion with user interaction

### Alternatives
- auto-rebuild vectors every app start

### Why not chosen
Too slow for the interactive app.

---

## 5.3 `README.md`

### Purpose
General setup and project introduction.

### Why it exists
New users need a quick entry point.

### Limitations
The README is broad and does not explain the KCET logic in enough depth.

---

## 5.4 `requirements.txt`

### Purpose
Lists dependencies.

### Why it exists
Ensures reproducible installation.

### Why dependencies matter
This project depends on:
- UI
- PDF parsing
- vector search
- LLM orchestration
- spreadsheet support

---

## 5.5 `data/information.md`

### Purpose
Notes about what belongs in the vector database folder.

### Why it exists
It helps organize source documents.

---

## 5.6 `data/kcet_cutoffs_template.csv`

### Purpose
Provides a sample structured cutoff format.

### Why it exists
Useful if the user has data in spreadsheet form.

---

## 5.7 `vector_db_dir/`

### Purpose
Persistent storage for embedded chunks.

### Why it exists
So the RAG chatbot does not need to rebuild embeddings every time.

### Contents
- Chroma sqlite storage
- embedded document chunks

---

## 5.8 `images/`

### Purpose
Holds image assets from the repo.

### Why it matters
Mostly for documentation or UI assets.

---

## 6. Tech Stack Analysis

### Streamlit
#### What it is
A Python framework for interactive web apps.

#### Why chosen
Fastest way to build a usable UI for filtering and chat.

#### Alternatives
- React
- Dash
- Flask templates

#### Why Streamlit fits
The app is form-heavy and interactive, not a complex SPA.

### FastAPI
#### What it is
A modern Python API framework.

#### Why chosen
Simple REST endpoint support and validation.

#### Alternatives
- Flask
- Django

#### Why FastAPI fits
Typed request models and clean API behavior.

### Pandas
#### What it is
Data manipulation library.

#### Why chosen
Ideal for tables, filtering, grouping, and sorting.

### PyPDF2
#### What it is
PDF text extraction library.

#### Why chosen
Lightweight and simple for text PDFs.

### Chroma
#### What it is
Vector database.

#### Why chosen
Easy local persistence and retrieval.

### LangChain
#### What it is
Orchestration framework for LLM workflows.

#### Why chosen
Connects retrieval, memory, prompt, and generation.

### HuggingFaceEmbeddings
#### What it is
Embedding model wrapper.

#### Why chosen
Works locally and integrates with LangChain.

### Groq / Ollama
#### What they are
LLM backends.

#### Why chosen
Supports hosted and local execution paths.

### FPDF
#### What it is
PDF generation library.

#### Why chosen
Useful for exporting transcripts or results.

---

## 7. Code Flow Walkthrough

### Example Flow: Rank Shortlist

User input:
- rank = 18000
- round = 3
- category = GM
- branch = computer science

Flow:
1. Streamlit reads the inputs
2. File uploads are cached
3. `find_kcet_matches` is called
4. cutoff rows are filtered
5. `branch_college_list` formats output
6. dataframe is shown

### Example Flow: Ask Questions

User input:
- “What does this PDF say about round 3?”

Flow:
1. input goes to `answer_question`
2. relevance check runs
3. retrieval gets top chunks
4. LLM generates answer
5. sources are shown

---

## 8. API Analysis

### `POST /chat`

#### Method
POST

#### Input
JSON body with `message`

#### Output
JSON response with `response`

#### Use case
External clients can ask the chatbot questions programmatically.

#### Risks
- untrusted text input
- weak context causing generic answers
- model latency

---

## 9. Interview Questions

### On Architecture
- Why use both Streamlit and FastAPI?
- Why use RAG instead of pure prompting?
- Why not use only CSV input?

### On Parsing
- How do you parse KEA PDFs?
- What issues happen with PDF extraction?
- How do you handle wrapped branch names?

### On Filtering
- Why is `GM` exact?
- Why are some categories expanded?
- Why use round windows?

### On Retrieval
- What is Chroma doing?
- Why check relevance before answering?
- How do you avoid hallucinations?

---

## 10. Possible Improvements

- OCR for scanned PDFs
- better parser for weird tables
- downloadable shortlist
- more polished compare cards
- district and quota filters
- branch family mapping table
- year-over-year cutoff charts
- exportable interview notes

---

## 11. Final Summary

This project is a hybrid admission support system:
- structured KCET shortlist engine
- PDF parser
- RAG chatbot
- comparison tool
- branch browser

It is built with a practical layered architecture so that students can search college cutoffs more easily and ask questions without manually reading every PDF.

---

## 12. Deep Function Walkthrough

This section explains the most important functions in a much more detailed way.

### 12.1 `remove_emojis(text)`

#### What it does
Removes emoji characters from a string before displaying or storing it.

#### Why it exists
Some model outputs or copied texts can include emojis, and those may not fit the professional tone of an admission assistant.

#### Why implemented this way
The function uses a regular expression over Unicode emoji ranges. That is a simple and lightweight way to clean the text without relying on an external library.

#### Alternative approaches
- use a dedicated emoji library
- sanitize only at render time

#### Why this approach is good
- fast
- local
- easy to understand

#### Disadvantages
- manual regex lists can miss new emoji ranges

#### Example input
`"Good luck 🎓 with your admission!"`

#### Example output
`"Good luck  with your admission!"`

#### Interview questions
- Why would you remove emojis from chatbot text?
- Could this impact user experience?

---

### 12.2 `setup_vectorstore()`

#### What it does
Loads the persistent Chroma vector database from disk.

#### Why it exists
The chatbot needs a memory of previously indexed documents. The vector store is that memory.

#### Why it is important
Without this step, the RAG chatbot would have no retrieval layer and would need to depend only on the language model.

#### Why Chroma is used here
It gives local persistence and integrates well with LangChain.

#### Alternative approaches
- FAISS
- Pinecone
- Weaviate

#### Why alternatives were not chosen
- FAISS is local but less convenient for persistence workflows
- Pinecone is cloud-based and less ideal for a simple local student project
- Weaviate is more feature-rich than necessary here

#### Potential errors
- missing vector directory
- corrupted embeddings
- dependency mismatch

#### Example behavior
If `vector_db_dir` contains embedded document chunks, the function returns a usable retriever-ready vector store.

---

### 12.3 `load_uploaded_pdfs(uploaded_files)`

#### What it does
Reads each uploaded PDF and extracts text page by page.

#### Why it exists
The project needs PDF text for both:
- cutoff parsing
- question answering

#### Execution flow
1. loop through uploaded files
2. open each file with `PdfReader`
3. extract text from every page
4. combine page text
5. store source and extracted text in a document record

#### Why page-level extraction matters
Some PDFs contain useful text on one page and tables on another. Preserving pages helps debugging and future enhancements like OCR or page citation.

#### Alternative approaches
- `pdfplumber`
- Apache Tika
- OCR-based extraction for image PDFs

#### Why PyPDF2 was chosen
- simple
- already available
- good enough for text-based PDFs

#### Limitations
- scanned PDFs may return little or no text
- complex table layouts can extract poorly

#### Example output shape
```python
[
  {
    "source": "PROF_CODE_E_R_11092025english.pdf",
    "text": "full extracted text...",
    "pages": ["page1 text", "page2 text"]
  }
]
```

---

### 12.4 `infer_round_number(round_text, source_name)`

#### What it does
Infers round number from the page title or filename.

#### Why it exists
Many cutoff PDFs do not store the round in a clean metadata field.

#### How it works
It checks:
- explicit words like `FIRST`, `SECOND`, `THIRD`
- `ROUND 1`, `ROUND 2`, `ROUND 3`
- filename hints like `1109`, `3008`, `R1`

#### Why fallback logic is needed
Some PDFs have vague or inconsistent text in the title section. File naming often gives a useful clue.

#### Alternatives
- manual round tagging
- OCR-based metadata extraction

#### Why this method is practical
It is simple and works on the kind of KEA files the project uses.

#### Example
Filename: `PROF_CODE_E_R_11092025english.pdf`

Likely result:
`3`

#### Potential issue
If the naming convention changes, the fallback mapping may need updates.

---

### 12.5 `normalize_category_query(category)`

#### What it does
Turns user input into a list of valid category codes that can be matched against the parsed dataframe.

#### Why it exists
Users type categories in a simplified form, but the actual PDF uses variant category codes.

#### Example
User types:
`SC`

Data may contain:
- `SCG`
- `SCK`
- `SCR`

This function helps bridge that gap.

#### Important design decision
`GM` stays exact.

That means:
- typing `GM` matches only `GM`
- it does not automatically expand to `GMK` or `GMR`

#### Why exact GM is important
The user specifically requested strict handling for general merit. This prevents category drift and avoids confusing results.

#### Alternative
Expand every category family automatically.

#### Why not chosen
That would be too broad for `GM` and would violate user expectations.

#### Example outputs
- input: `gm` -> `["GM"]`
- input: `sc` -> `["SCG", "SCK", "SCR"]`
- input: `2a` -> `["2AG", "2AK", "2AR"]`

#### Interview questions
- Why is category normalization necessary?
- Why is `GM` treated differently from `SC` or `ST`?

---

### 12.6 `clean_branch_name(branch)`

#### What it does
Removes parentheses and extra spaces from branch names.

#### Why it exists
Branch titles in cutoff PDFs are often too verbose and inconsistent.

#### Example
Input:
`"Computer Science and Engineering (Artificial Intelligence and Machine Learning)"`

Output:
`"Computer Science and Engineering"`

#### Why this helps
Students care about branch families more than the exact formatting of every PDF line.

#### Alternative
Show the original text exactly as parsed.

#### Why not chosen
The output would be noisy and harder to understand.

---

### 12.7 `normalize_branch_query(branch)`

#### What it does
Creates a normalized form of the branch filter for searching.

#### Why it exists
User input can be lowercase, abbreviated, or contain punctuation. Normalization makes search more robust.

#### Example
Input:
`"computer science"`

Normalized:
`"computer science"`

Input:
`"Computer Science & Engineering (AI/ML)"`

Normalized:
`"computer science and engineering ai ml"`

#### Why this is helpful
The search becomes tolerant to naming variation.

---

### 12.8 `branch_family_name(branch)`

#### What it does
Maps many noisy branch labels into a cleaner family label.

#### Why it exists
Cutoff PDFs may list different descriptive variants for the same broad branch area.

#### Example mappings
- anything containing `computer science` -> `Computer Science and Engineering`
- anything containing `information science` -> `Information Science and Engineering`
- anything containing `artificial intelligence`, `machine learning`, or `data science` -> `Artificial Intelligence and Data Science`

#### Why this design helps users
Instead of seeing 10 slightly different variants, the user sees one clean branch family.

#### Alternative
Keep separate labels for every PDF variant.

#### Why not chosen
That would create duplicate-looking rows and reduce readability.

---

### 12.9 `extract_cutoff_rows_from_page(page_text, source_name)`

#### What it does
Parses a text page from a KEA-style cutoff PDF and turns it into structured rows.

#### Why it is the heart of the project
This is the function that converts unstructured PDF text into usable tabular data.

#### Detailed execution flow
1. detect round text from the page title
2. detect seat type
3. detect year
4. split page into lines
5. find college section starts
6. parse each college section separately
7. extract college code and college name
8. locate the `Course Name` header
9. detect category codes
10. collect branch names and cutoff values
11. build row dictionaries

#### Why section splitting matters
One page may contain multiple colleges. If you read the page as one block, you can mix unrelated rows together.

#### Why header parsing matters
The category columns determine how the numeric values should be interpreted.

#### Why row reconstruction matters
The PDF may split a branch name across lines or merge values into one line.

#### Example output row
```python
{
  "college_code": "E001",
  "college_name": "University of Visvesvaraya College of Engineering",
  "course_name": "COMPUTER SCIENCE AND ENGINEERING",
  "branch": "COMPUTER SCIENCE AND ENGINEERING",
  "round": 3,
  "round_text": "THIRD",
  "seat_type": "All Seats",
  "category": "GM",
  "cutoff_rank": 19055.0,
  "year": 2025,
  "source": "PROF_CODE_E_R_11092025english.pdf"
}
```

#### Alternatives
- OCR on the full page image
- specialized table extraction tools

#### Why the current method was chosen
The PDFs are text-based enough to justify a direct text parser first. OCR is a fallback, not the first step.

#### Risks
- malformed rows
- merged tokens
- missing pages
- skipped colleges

#### Interview questions
- Why did you not use OCR first?
- How do you handle multiple colleges per PDF page?
- What happens if a branch name wraps across lines?

---

### 12.10 `load_uploaded_cutoff_data(uploaded_files)`

#### What it does
Converts all uploaded files into one unified dataframe.

#### Why it exists
The rest of the app should not care whether the original file was PDF, CSV, or Excel.

#### Why this abstraction is useful
It creates a single source of truth for shortlist filtering.

#### Execution flow
1. inspect file extension
2. parse accordingly
3. validate expected columns if tabular file
4. parse PDF rows if PDF
5. combine all records into one dataframe

#### Why this is good architecture
The UI only interacts with one dataframe, which simplifies everything downstream.

---

### 12.11 `build_uploaded_pdf_vectorstore(uploaded_files)`

#### What it does
Creates a temporary vector database from uploaded PDFs.

#### Why it exists
The chat mode should be able to answer from user-uploaded PDFs immediately.

#### Why chunking is used
LLMs and retrievers work better on smaller text chunks than on giant full-document blobs.

#### Why overlap is used
Overlap preserves context across chunk boundaries.

#### Alternative
Index the entire PDF as one document.

#### Why not chosen
Large documents would retrieve poorly and answer less precisely.

---

### 12.12 `round_window(round_no)`

#### What it does
Returns the admissible upper rank window for a given round.

#### Why it exists
Different rounds have different practical cutoff behavior.

#### Example
If round is `1`, the app searches up to `rank + 3000`.

#### Why this is better than a fixed delta
It makes the shortlist reflect round-based trends instead of one rigid assumption.

#### Alternative
One universal tolerance.

#### Why not chosen
Round 1 and round 3 behave differently in real admissions data.

---

### 12.13 `find_kcet_matches(...)`

#### What it does
Filters the structured data using the student’s inputs.

#### Why it matters
This is the actual college shortlist logic.

#### Filter order
1. rank window
2. round
3. category
4. branch

#### Why the order matters
It reduces the dataframe step by step and avoids unnecessary processing.

#### Example
If rank = 18000 and tolerance = 2800:
The app considers rows from `18000` to `20800`.

Then it filters:
- round = 3
- category = GM
- branch = computer science

#### Potential issue
If the parser itself extracted bad values, even a perfect filter will still return poor output.

#### Interview questions
- Why use numeric filtering before display?
- Why not just use the LLM to answer from the PDF?

---

### 12.14 `branch_college_list(df)`

#### What it does
Converts matching raw rows into a clean shortlist table.

#### Why it exists
Students should not see the raw parser internals. They should see a polished table.

#### Why deduping is needed
The same college-branch-round-category combination may appear more than once across uploaded files or parsed sections.

#### Why output is simplified
Too many columns make the shortlist harder to interpret.

#### Example columns
- college name
- branch
- round
- category
- cutoff
- year

#### Interview question
- Why did you choose these columns and not all parsed fields?

---

### 12.15 `build_data_summary(df)`

#### What it does
Produces a quick health summary of the loaded data.

#### Why it exists
Users need immediate feedback that the upload worked.

#### Example summary
- Colleges: 120
- Branches: 18
- Rounds: 1, 2, 3
- Years: 2025

#### Why summaries are helpful
They prevent silent failure and improve trust in the upload pipeline.

---

### 12.16 `sorted_unique(values)`

#### What it does
Returns unique text values in a clean list.

#### Why it exists
Used for dropdowns and multiselects.

#### Why not just use `set()`
`set()` loses order and can produce messy UI ordering.

---

### 12.17 `setup_llm()`

#### What it does
Chooses Groq or Ollama as the language model backend.

#### Why it exists
Supports both hosted and local execution.

#### Why this is practical
The project can still run locally even if the user does not have a cloud API key.

---

### 12.18 `chat_chain(vectorstore)`

#### What it does
Builds the retrieval-based conversation chain.

#### Why it exists
This is the RAG engine.

#### What it combines
- retriever
- memory
- prompt
- LLM

#### Why conversational memory matters
It allows follow-up questions to be answered in context.

#### Example
User:
“What about round 3?”

Then:
“What branches are available there?”

The conversation chain helps maintain continuity.

---

### 12.19 `answer_question(...)`

#### What it does
Checks relevance, then runs the conversation chain.

#### Why it exists
It prevents the LLM from answering when the retrieval context is weak.

#### Why this is a good safety layer
It reduces random or unsupported answers.

#### Alternative
Always answer regardless of context.

#### Why not chosen
That would increase hallucination risk.

---

## 13. Example End-to-End Scenarios

### Scenario A: Student wants a shortlist

Input:
- Rank: 18000
- Round: 3
- Category: GM
- Branch: computer science

What happens:
- upload data is parsed
- rows are filtered
- cleaned branch labels are shown
- shortlist table appears

Why this is helpful:
- the student quickly sees realistic options
- no manual PDF searching is needed

### Scenario B: Student compares colleges

Input:
- selects 3 colleges

What happens:
- each college becomes a side-by-side card
- best matching cutoff is shown
- the rest is hidden behind expanders

Why this is helpful:
- easier visual comparison

### Scenario C: Student asks a question

Input:
- “What does the PDF say about seat type?”

What happens:
- relevant chunks are retrieved
- the LLM answers with context
- source references are displayed

Why this is helpful:
- the student can query the documents conversationally

---

## 14. Interview Preparation Points

### What to say about the project
This project is a KCET cutoff assistant that combines structured filtering and RAG-based document question answering.

### Why the project is strong
- solves a real student problem
- uses multiple technologies meaningfully
- has both search and chat
- handles messy PDFs

### What the interviewer may ask
- How does PDF parsing work?
- Why did you use Chroma?
- Why not use only the LLM?
- How did you normalize category and branch names?
- Why is compare mode side-by-side?

### Best answer style
Explain:
- the problem
- the technical choices
- the tradeoffs
- the final user benefit

---

## 15. Final Technical Takeaway

This project is not just “a chatbot.”

It is actually:
- a PDF parsing pipeline
- a structured admission shortlist engine
- a comparison interface
- a document retrieval assistant

That combination is what makes it useful in the real world.
