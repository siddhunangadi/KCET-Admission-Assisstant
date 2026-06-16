# KCET College Finder

KCET College Finder is a Streamlit app for students who want to compare KCET cutoff PDFs and quickly shortlist possible colleges and branches by rank, round, category, and branch.

The app is designed for a real student workflow:
- upload cutoff PDFs from multiple years
- choose rank, round, category, and branch
- compare the same college across years
- ask questions in natural language
- see the results in a table instead of reading every PDF manually

---

## What This Project Does

- parses uploaded KCET cutoff PDFs, CSVs, or Excel files
- extracts college name, branch, category, cutoff rank, round, and year
- groups matching rows across years
- shows a year-wise comparison table
- supports a chat-style interface for follow-up questions
- uses Mistral for natural-language responses

---

## Main Idea

The project is not trying to predict admissions.

It uses the uploaded cutoff data as the source of truth and answers questions like:
- which colleges are possible for a given rank?
- how did the cutoff change from 2024 to 2025?
- what are the probable colleges for a branch like CSE or ISE?
- which branch options are available in a college?

---

## Features

- multi-file upload support
- PDF parsing for KEA-style cutoff documents
- CSV / Excel support if cutoff data is already structured
- rank, round, category, and branch filtering
- year-by-year comparison table
- optional college-specific lookup
- Mistral-powered chatbot
- local Streamlit UI

---

## Project Flow

1. Upload cutoff files in the sidebar.
2. The app parses the files and loads them into a dataframe.
3. The user fills the KCET intake form:
   - rank
   - round
   - category
   - branch
4. The app filters matching cutoff rows.
5. Results are grouped and shown as a comparison table.
6. If the user asks a natural-language question, Mistral uses the saved profile and cutoff context to answer.

---

## Example Output Logic

For the same college and branch, the app groups rows like:
- college name
- branch
- round
- category

Then it shows the cutoff columns by year, for example:
- 2023
- 2024
- 2025

It also includes:
- `best_cutoff`
- `trend`
- `years_available`

---

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Mistral
- **Embeddings / RAG support**: LangChain
- **Vector store**: Pinecone
- **PDF parsing**: PyPDF2
- **Data handling**: Pandas
- **API layer**: FastAPI

---

## Requirements

- Python 3.9+
- a virtual environment is recommended
- Mistral API key
- Pinecone API key and index name

> The current code still initializes Pinecone-backed retrieval, so the Pinecone values must be present in `.env`.

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Create `.env`

Create a `.env` file in the project root and add:

```env
MISTRAL_API_KEY=your_mistral_key_here
MISTRAL_MODEL=mistral-small-2506

PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=your_index_name
PINECONE_NAMESPACE=
```

If you do not want to use a namespace, leave `PINECONE_NAMESPACE` blank.

### 4. Run the app

```bash
python -m streamlit run main.py
```

---

## How To Use

1. Open the app in your browser.
2. Upload the cutoff PDFs you want to compare.
3. Fill in the intake form.
4. Ask a question like:
   - `give me colleges`
   - `show all years for Sri Venkateshwara College of Engineering in CSE`
   - `compare Dayananda Sagar University CSE cutoffs`

---

## Input File Notes

The app works best with KEA-style cutoff PDFs.

If the PDF is structured like a compact allotment report, the parser tries to infer:
- year
- college name
- branch
- category
- cutoff rank
- round

If the PDF is not structured well, the app may still show the raw extracted text, but the comparison table can be incomplete.

---

## Important Logic

The shortlist is built from:
- student rank
- round
- category
- branch
- cutoff rank from the uploaded files

For year comparison:
- the same college + branch + round + category is grouped together
- cutoff values are shown year by year
- the table includes all available years found in the uploaded files

---

## Project Structure

```text
main.py                 # Streamlit app + parsing + chat flow
vectorize_documents.py   # optional document vectorization helper
requirements.txt         # Python dependencies
data/                   # sample/support documents
vector_db_dir/          # local vector database storage
config.json             # legacy config support
```

---

## Troubleshooting

### 1. Mistral not configured

Make sure `.env` contains:

```env
MISTRAL_API_KEY=...
```

### 2. Pinecone namespace error

If you see a namespace error, leave `PINECONE_NAMESPACE` blank in `.env`.

### 3. No rows parsed from PDF

Try a different cutoff PDF or a more structured KEA document.

### 4. Table looks empty

Make sure:
- the right files are uploaded
- the rank, round, category, and branch are filled correctly
- the branch is broad enough if you want all colleges

---

## Goal Of The Project

The goal is to help KCET students:
- avoid manually searching many cutoff PDFs
- compare college cutoffs across years
- shortlist realistic options faster
- get answers in a natural-language style

---

## Future Improvements

- stronger KEA PDF parsing for more layouts
- better branch alias handling
- exact college-specific filters and aliases
- clearer multi-year trend charts
- predictions for future cutoff ranges

---

## License

This project currently does not declare a separate license. Add one if you plan to share or publish it.
