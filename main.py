# Change "College Name" to the actual college name you are building the chatbot for.
# Throughout the code, replace "College Name" with the actual college name.
# This is a FastAPI backend combined with a Streamlit frontend for a chatbot using LangChain and Groq LLM.
# You should have a config.json file in the same directory with your GROQ_API_KEY.
# Make sure to install all required packages:
# pip install -r requirements.txt and the requirements.txt should include:
# fastapi, uvicorn, pydantic, streamlit, langchain, langchain-huggingface, langchain-pinecone, langchain-groq, fpdf
# To run the project:
# - Run streamlit: streamlit run main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
from io import BytesIO
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF  # Using the original fpdf package
from datetime import datetime
from PyPDF2 import PdfReader

try:
    from langchain_pinecone import PineconeVectorStore
except Exception:
    PineconeVectorStore = None

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None

try:
    from mistralai import Mistral
except Exception:
    Mistral = None

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

working_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(working_dir, ".env"))
config_path = os.path.join(working_dir, "config.json")
config_data = {}
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        try:
            config_data = json.load(config_file)
        except json.JSONDecodeError:
            config_data = {}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip() or config_data.get("GROQ_API_KEY", "").strip()
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip() or config_data.get("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest").strip()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "").strip() or config_data.get("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "").strip() or config_data.get("PINECONE_INDEX_NAME", "").strip()
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "").strip() or config_data.get("PINECONE_NAMESPACE", "").strip()
if PINECONE_NAMESPACE in {"__default__", "default"}:
    PINECONE_NAMESPACE = ""

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if PINECONE_INDEX_NAME:
    os.environ["PINECONE_INDEX_NAME"] = PINECONE_INDEX_NAME
if PINECONE_NAMESPACE:
    os.environ["PINECONE_NAMESPACE"] = PINECONE_NAMESPACE
else:
    os.environ.pop("PINECONE_NAMESPACE", None)

# FastAPI app and Pydantic model for API input
app = FastAPI()

class MessageRequest(BaseModel):
    message: str

# FastAPI route for chatbot
@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message

    # Setup vectorstore (same as Streamlit code)
    vectorstore = setup_vectorstore()

    # Setup the conversational chain (same as Streamlit code)
    conversational_chain = chat_chain(vectorstore)

    # Check for sensitive topics
    if contains_sensitive_topics(message):
        response = "It seems you may be asking questions outside my context, please ask questions related to College Name only."
    else:
        response, _sources = answer_question(vectorstore, conversational_chain, message, history_messages=[], cutoff_df=load_cutoff_data())

    return {"response": response}

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a KCET college shortlist assistant.

Your job is to ask the student for missing details first, then use only the uploaded cutoff data to answer.

Always collect these details before giving a shortlist:
- Rank
- Category
- Round
- Branch or preferred course
- Optional college preference if the student has one

Rules:
- If any required detail is missing, ask only for the missing details.
- Do not guess colleges or cutoffs.
- Do not answer with generic advice when the user is asking for KCET seat availability.
- Once enough details are available, return a clean shortlist from the uploaded data.
- Keep the tone student-friendly and practical.
"""

DEFAULT_NEGATIVE_PROMPT = """
- Do **NOT** provide any information that is **not supported by verified College Name data** or the provided system context.
- Do **NOT** imply you are an **employee, representative, agent, or official spokesperson** of College Name.
- Do **NOT** fabricate or invent College Name **services, features, pricing, policies, internal processes, or proprietary details**.
- Do **NOT** offer **legal, financial, medical, or other unrelated professional advice** outside College Name's domain.
- Do **NOT** respond to topics **outside College Name's scope**; instead, politely state that the relevant data is not available.
- Do **NOT** guess or assume **confidential, internal, or sensitive business information** about College Name.
- Do **NOT** generate speculative, generic, or hypothetical business advice that is **not grounded in College Name's verified information**.
- Do **NOT** use, cite, or reference **external sources, external knowledge, or outside databases** beyond the authorized College Name context.
- Do **NOT** insert personal opinions, assumptions, unfounded claims, or subjective judgments.
- Do **NOT** mislead the user with unsupported or speculative responses.
- Do **NOT** use an unprofessional, casual, or overly familiar tone; maintain professionalism at all times.
"""

def contains_sensitive_topics(question):
    sensitive_keywords = [
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)

def setup_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    if PineconeVectorStore is None or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise RuntimeError(
            "Pinecone is required for this project. Please set PINECONE_API_KEY and PINECONE_INDEX_NAME."
        )

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE or None,
    )

@st.cache_resource(show_spinner=False)
def get_base_vectorstore():
    return setup_vectorstore()

def setup_uploaded_pdf_vectorstore(uploaded_files):
    pdf_docs = load_uploaded_pdfs(uploaded_files)
    if not pdf_docs:
        return None

    if PineconeVectorStore is None or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise RuntimeError(
            "Pinecone is required for uploaded PDF retrieval. Please set PINECONE_API_KEY and PINECONE_INDEX_NAME."
        )

    embeddings = HuggingFaceEmbeddings()
    documents = []
    for item in pdf_docs:
        documents.append({
            "page_content": item["text"],
            "metadata": {"source": item["source"]},
        })

    try:
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        langchain_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in documents]
        chunks = splitter.split_documents(langchain_docs)
        index = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=(PINECONE_NAMESPACE or None),
        )
        index.add_documents(chunks)
        return index
    except Exception:
        return None

def load_cutoff_data():
    cutoff_path = os.path.join(working_dir, "data", "kcet_cutoffs.csv")
    template_path = os.path.join(working_dir, "data", "kcet_cutoffs_template.csv")

    if os.path.exists(cutoff_path):
        path = cutoff_path
    elif os.path.exists(template_path):
        path = template_path
    else:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        expected = {"college_name", "branch", "round", "category", "cutoff_rank", "year"}
        if not expected.issubset(set(df.columns)):
            return pd.DataFrame()
        df["cutoff_rank"] = pd.to_numeric(df["cutoff_rank"], errors="coerce")
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        return df.dropna(subset=["college_name", "branch", "round", "category", "cutoff_rank"])
    except Exception:
        return pd.DataFrame()

def load_cutoff_file(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    expected = {"college_name", "branch", "round", "category", "cutoff_rank", "year"}
    if not expected.issubset(set(df.columns)):
        return pd.DataFrame()

    df["cutoff_rank"] = pd.to_numeric(df["cutoff_rank"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df.dropna(subset=["college_name", "branch", "round", "category", "cutoff_rank"])

def load_uploaded_pdfs(uploaded_files):
    documents = []
    if not uploaded_files:
        return documents

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            combined = "\n".join(pages).strip()
            if combined:
                documents.append({
                    "source": uploaded_file.name,
                    "text": combined,
                    "pages": pages,
                })
        except Exception:
            continue
    return documents

def infer_round_number(round_text, source_name=""):
    text = f"{round_text} {source_name}".upper()

    if re.search(r"\bTHIRD\b|\bROUND\s*3\b|\bR3\b|\b3RD\b", text):
        return 3
    if re.search(r"\bSECOND\b|\bROUND\s*2\b|\bR2\b|\b2ND\b", text):
        return 2
    if re.search(r"\bFIRST\b|\bROUND\s*1\b|\bR1\b|\b1ST\b", text):
        return 1

    # File name/date hints used as fallback for KEA-style PDFs
    if re.search(r"1109", source_name):
        return 3
    if re.search(r"3008", source_name):
        return 2
    if re.search(r"r1", source_name, re.I):
        return 1

    return 1

def normalize_category_query(category):
    if not category:
        return []

    cat = category.strip().upper().replace(" ", "")
    aliases = {
        "GM": ["GM"],
        "GENERAL": ["GM"],
        "GMR": ["GMR"],
        "GMK": ["GMK"],
        "SC": ["SCG", "SCK", "SCR"],
        "SCG": ["SCG", "SCK", "SCR"],
        "SCK": ["SCK", "SCG", "SCR"],
        "SCR": ["SCR", "SCG", "SCK"],
        "ST": ["STG", "STK", "STR"],
        "STG": ["STG", "STK", "STR"],
        "STK": ["STK", "STG", "STR"],
        "STR": ["STR", "STG", "STK"],
        "1G": ["1G", "1K", "1R"],
        "1K": ["1K", "1G", "1R"],
        "1R": ["1R", "1G", "1K"],
        "2A": ["2AG", "2AK", "2AR"],
        "2AG": ["2AG", "2AK", "2AR"],
        "2AK": ["2AK", "2AG", "2AR"],
        "2AR": ["2AR", "2AG", "2AK"],
        "2B": ["2BG", "2BK", "2BR"],
        "2BG": ["2BG", "2BK", "2BR"],
        "2BK": ["2BK", "2BG", "2BR"],
        "2BR": ["2BR", "2BG", "2BK"],
        "3A": ["3AG", "3AK", "3AR"],
        "3AG": ["3AG", "3AK", "3AR"],
        "3AK": ["3AK", "3AG", "3AR"],
        "3AR": ["3AR", "3AG", "3AK"],
        "3B": ["3BG", "3BK", "3BR"],
        "3BG": ["3BG", "3BK", "3BR"],
        "3BK": ["3BK", "3BG", "3BR"],
        "3BR": ["3BR", "3BG", "3BK"],
    }
    return aliases.get(cat, [cat])

def clean_branch_name(branch):
    if pd.isna(branch):
        return ""
    text = str(branch)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_course_name(course_name):
    if pd.isna(course_name):
        return ""
    text = str(course_name)
    text = text.replace("\n", " ")
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)
    text = re.sub(r"[|/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -,&")

def normalize_branch_query(branch):
    if not branch:
        return ""
    text = clean_branch_name(branch).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def branch_family_name(branch):
    text = clean_course_name(branch)
    lowered = text.lower()
    patterns = [
        (r"\bcomputer science and engineering\b.*\b(internet of things|iot).*blockchain", "Computer Science and Engineering (Internet of Things and Cyber Security Including Blockchain Technology)"),
        (r"\bcomputer science and engineering\b.*\b(cyber security)\b", "Computer Science and Engineering (Cyber Security)"),
        (r"\bcomputer science and engineering\b.*\b(artificial intelligence and machine learning|ai/ml|aiml)\b", "Computer Science and Engineering (Artificial Intelligence and Machine Learning)"),
        (r"\bcomputer science and engineering\b.*\b(artificial intelligence|ai)\b", "Computer Science and Engineering (Artificial Intelligence)"),
        (r"\bcomputer science and engineering\b.*\b(data science|dat)\b", "Computer Science and Engineering (Data Science)"),
        (r"\bcomputer science and business systems\b", "Computer Science and Business Systems"),
        (r"\bcomputer science and design\b", "Computer Science and Design"),
        (r"\bcomputer science and technology\b", "Computer Science and Technology (Exclusively for Differently Abled)"),
        (r"\bartificial intelligence and machine learning\b|^\s*aiml\s*$", "Artificial Intelligence and Machine Learning"),
        (r"\bartificial intelligence and data science\b|^\s*aids\s*$", "Artificial Intelligence and Data Science"),
        (r"\baerospace engineering\b", "Aerospace Engineering"),
        (r"\baeronautical engineering\b", "Aeronautical Engineering"),
        (r"\bagriculture engineering\b", "Agriculture Engineering"),
        (r"\bautomobile engineering\b", "Automobile Engineering"),
        (r"\bbio-?medical engineering\b", "Bio-Medical Engineering"),
        (r"\bbiotechnology\b", "Biotechnology"),
        (r"\bchemical engineering\b", "Chemical Engineering"),
        (r"\bceramics and cement engineering\b", "Ceramics and Cement Engineering"),
        (r"\bcivil engineering \(kannada medium\)\b", "Civil Engineering (Kannada Medium)"),
        (r"\bcivil engineering\b", "Civil Engineering"),
        (r"\bconstruction technology and management\b", "Construction Technology and Management"),
        (r"\bdata sciences?\b", "Data Sciences"),
        (r"\belectrical and electronics engineering\b", "Electrical and Electronics Engineering"),
        (r"\belectronics and computer engineering\b", "Electronics and Computer Engineering"),
        (r"\belectronics and communication engineering\b.*\b(advanced communication technology)\b", "Electronics and Communication Engineering (Advanced Communication Technology)"),
        (r"\belectronics and communication engineering\b.*\b(vlsi design and technology)\b", "Electronics and Communication Engineering (VLSI Design and Technology)"),
        (r"\belectronics and communication engineering\b", "Electronics and Communication Engineering"),
        (r"\belectronics and instrumentation engineering\b", "Electronics and Instrumentation Engineering"),
        (r"\belectronics and telecommunication engineering\b", "Electronics and Telecommunication Engineering"),
        (r"\belectronics engineering\b.*\b(vlsi design and technology)\b", "Electronics Engineering (VLSI Design and Technology)"),
        (r"\benvironmental engineering\b", "Environmental Engineering"),
        (r"\bindustrial and production engineering\b", "Industrial and Production Engineering"),
        (r"\bindustrial engineering and management\b", "Industrial Engineering and Management"),
        (r"\binformation science and engineering\b", "Information Science and Engineering"),
        (r"\bmechanical engineering\b", "Mechanical Engineering"),
        (r"\bmedical electronics engineering\b", "Medical Electronics Engineering"),
        (r"\bmining engineering\b", "Mining Engineering"),
        (r"\bpolymer science and technology\b", "Polymer Science and Technology"),
        (r"\brobotics and artificial intelligence\b", "Robotics and Artificial Intelligence"),
        (r"\bsilk technology\b", "Silk Technology"),
        (r"\btextiles technology\b", "Textiles Technology"),
    ]
    normalized = normalize_branch_query(text)
    for pattern, canonical in patterns:
        if re.search(pattern, normalized, re.I):
            return canonical
    return None

def canonical_branch_options(branch_series):
    canonical = {}
    for raw_branch in branch_series.dropna().tolist():
        display_branch = branch_family_name(raw_branch)
        if not display_branch:
            continue
        key = normalize_branch_query(display_branch)
        if key and key not in canonical:
            canonical[key] = display_branch
    approved_order = [
        "Aerospace Engineering",
        "Aeronautical Engineering",
        "Agriculture Engineering",
        "Artificial Intelligence and Data Science",
        "Artificial Intelligence and Machine Learning",
        "Automobile Engineering",
        "Bio-Medical Engineering",
        "Biotechnology",
        "Chemical Engineering",
        "Ceramics and Cement Engineering",
        "Civil Engineering",
        "Civil Engineering (Kannada Medium)",
        "Computer Science and Engineering",
        "Computer Science and Engineering (Artificial Intelligence)",
        "Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
        "Computer Science and Engineering (Cyber Security)",
        "Computer Science and Engineering (Data Science)",
        "Computer Science and Engineering (Internet of Things and Cyber Security Including Blockchain Technology)",
        "Computer Science and Business Systems",
        "Computer Science and Design",
        "Computer Science and Technology (Exclusively for Differently Abled)",
        "Construction Technology and Management",
        "Data Sciences",
        "Electrical and Electronics Engineering",
        "Electronics and Computer Engineering",
        "Electronics and Communication Engineering",
        "Electronics and Communication Engineering (Advanced Communication Technology)",
        "Electronics and Communication Engineering (VLSI Design and Technology)",
        "Electronics and Instrumentation Engineering",
        "Electronics and Telecommunication Engineering",
        "Electronics Engineering (VLSI Design and Technology)",
        "Environmental Engineering",
        "Industrial and Production Engineering",
        "Industrial Engineering and Management",
        "Information Science and Engineering",
        "Mechanical Engineering",
        "Medical Electronics Engineering",
        "Mining Engineering",
        "Polymer Science and Technology",
        "Robotics and Artificial Intelligence",
        "Silk Technology",
        "Textiles Technology",
    ]
    approved_set = {normalize_branch_query(name): name for name in approved_order}
    ordered = ["All"]
    for branch_name in approved_order:
        key = normalize_branch_query(branch_name)
        if key in canonical or key in approved_set:
            ordered.append(approved_set[key])
    return ordered

def file_signature(uploaded_files):
    if not uploaded_files:
        return ()
    return tuple((f.name, getattr(f, "size", None)) for f in uploaded_files)

def _extract_fixed_columns(compact_values, expected_count):
    tokens = []

    def helper(idx):
        if len(tokens) == expected_count and idx == len(compact_values):
            return True
        if len(tokens) >= expected_count or idx >= len(compact_values):
            return False

        remaining_slots = expected_count - len(tokens)
        remaining_chars = len(compact_values) - idx
        if remaining_chars < remaining_slots:
            return False

        if compact_values.startswith("--", idx):
            tokens.append("--")
            if helper(idx + 2):
                return True
            tokens.pop()

        max_len = min(8, len(compact_values) - idx)
        for end in range(idx + 1, idx + max_len + 1):
            piece = compact_values[idx:end]
            if not re.fullmatch(r"\d+(?:\.\d+)?", piece):
                continue
            tokens.append(piece)
            if helper(end):
                return True
            tokens.pop()
        return False

    if helper(0):
        return tokens
    return []

def infer_year_from_source(page_text, source_name):
    text = f"{page_text} {source_name}".lower()
    year_matches = re.findall(r"\b(20\d{2})\b", text)
    preferred_years = [int(y) for y in year_matches if 2020 <= int(y) <= 2026]
    if preferred_years:
        return max(preferred_years)

    # Common KCET/PDF naming hints
    if "2026" in text:
        return 2026
    if "2025" in text:
        return 2025
    if "2024" in text:
        return 2024
    if "2023" in text:
        return 2023

    return 2025

def extract_cutoff_rows_from_page(page_text, source_name):
    rows = []
    seen = set()
    round_match = re.search(r"UGCET-\d+\s+([A-Z ]+?)\s+CUT-OFF RANKS", page_text, re.I)
    round_text = round_match.group(1).strip() if round_match else ""

    seat_match = re.search(r"Seat Type:\s*(.+?)\s*Cut-Off Ranks", page_text, re.I)
    seat_type = seat_match.group(1).strip() if seat_match else ""

    year = infer_year_from_source(page_text, source_name)

    lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
    college_starts = []
    for idx, line in enumerate(lines):
        if line.startswith("College:"):
            college_starts.append(idx)

    if not college_starts:
        return rows

    for start_idx, college_idx in enumerate(college_starts):
        end_idx = college_starts[start_idx + 1] if start_idx + 1 < len(college_starts) else len(lines)
        section_lines = lines[college_idx:end_idx]
        section_text = "\n".join(section_lines)

        college_match = re.search(r"College:\s*(?:\((E\d{3})\)|([A-Z]\d{3}))\s*(.+)", section_text)
        if not college_match:
            continue
        college_code = (college_match.group(1) or college_match.group(2)).strip()
        college_name = college_match.group(3).split("\n", 1)[0].strip()
        round_no = infer_round_number(round_text, source_name)

        header_match = re.search(r"Course Name\s+(.+)", section_text)
        if not header_match:
            continue

        categories_raw = header_match.group(1).replace("GMGMK", "GM GMK")
        category_codes = re.findall(r"1G|1K|1R|2AG|2AK|2AR|2BG|2BK|2BR|3AG|3AK|3AR|3BG|3BK|3BR|GMK|GMP|GMR|NRI|OPN|OTH|SCG|SCK|SCR|STG|STK|STR|GM", categories_raw)
        if not category_codes:
            continue

        rows_start = None
        for idx, line in enumerate(section_lines):
            if line.startswith("Course Name"):
                rows_start = idx + 1
                break
        if rows_start is None:
            continue

        current_course = []
        current_values = []

        def flush_course():
            nonlocal current_course, current_values
            if not current_course or not current_values:
                current_course = []
                current_values = []
                return

            course_name = re.sub(r"\s+", " ", " ".join(current_course)).strip()
            course_name = clean_course_name(course_name)
            if not course_name:
                current_course = []
                current_values = []
                return
            branch_name = branch_family_name(course_name)
            if not branch_name:
                current_course = []
                current_values = []
                return

            compact = " ".join(current_values)
            tokens = re.findall(r"--|\d+(?:\.\d+)?", compact)
            if len(tokens) < len(category_codes):
                current_course = []
                current_values = []
                return

            tokens = tokens[:len(category_codes)]
            for category, value in zip(category_codes, tokens):
                if value == "--":
                    continue
                key = (college_code, college_name.lower(), course_name.lower(), category, value, year, round_text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "college_code": college_code,
                    "college_name": college_name,
                    "course_name": course_name,
                    "branch": branch_name,
                    "round": round_no,
                    "round_text": round_text,
                    "seat_type": seat_type,
                    "category": category,
                    "cutoff_rank": float(value),
                    "year": year,
                    "source": source_name,
                })
            current_course = []
            current_values = []

        for line in section_lines[rows_start:]:
            if not line or line.startswith("Generated on:"):
                continue
            if line.startswith("College:"):
                flush_course()
                break
            if line.startswith("Course Name"):
                flush_course()
                continue

            has_any_digits = bool(re.search(r"\d", line))
            has_any_dash = "--" in line

            if has_any_digits or has_any_dash:
                if not current_course:
                    split_at = re.search(r"(--|\d)", line)
                    if split_at:
                        course_part = line[:split_at.start()].strip()
                        if course_part:
                            current_course.append(course_part)
                        current_values.append(line[split_at.start():].strip())
                    else:
                        current_values.append(line)
                else:
                    current_values.append(line)
            else:
                if current_values:
                    flush_course()
                current_course.append(line)

        flush_course()
    return rows

def load_uploaded_cutoff_data(uploaded_files):
    rows = []
    if not uploaded_files:
        return pd.DataFrame()

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                expected = {"college_name", "branch", "round", "category", "cutoff_rank", "year"}
                if expected.issubset(set(df.columns)):
                    df["cutoff_rank"] = pd.to_numeric(df["cutoff_rank"], errors="coerce")
                    df["round"] = pd.to_numeric(df["round"], errors="coerce")
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")
                    df = df.dropna(subset=["college_name", "branch", "round", "category", "cutoff_rank"])
                    rows.extend(df.to_dict("records"))
                    continue

            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(uploaded_file)
                expected = {"college_name", "branch", "round", "category", "cutoff_rank", "year"}
                if expected.issubset(set(df.columns)):
                    df["cutoff_rank"] = pd.to_numeric(df["cutoff_rank"], errors="coerce")
                    df["round"] = pd.to_numeric(df["round"], errors="coerce")
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")
                    df = df.dropna(subset=["college_name", "branch", "round", "category", "cutoff_rank"])
                    rows.extend(df.to_dict("records"))
                    continue

            if filename.endswith(".pdf"):
                pdf_docs = load_uploaded_pdfs([uploaded_file])
                for item in pdf_docs:
                    for page_text in item.get("pages", []):
                        rows.extend(extract_cutoff_rows_from_page(page_text, item["source"]))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
    if "cutoff_rank" in df.columns:
        df["cutoff_rank"] = pd.to_numeric(df["cutoff_rank"], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df.dropna(subset=["college_name", "branch", "round", "category", "cutoff_rank"])

def build_uploaded_pdf_vectorstore(uploaded_files):
    return setup_uploaded_pdf_vectorstore(uploaded_files)

def round_window(round_no):
    if round_no == 1:
        return 3000
    if round_no == 2:
        return 2800
    if round_no == 3:
        return 2800
    return 2800

def find_kcet_matches(df, rank, round_no=None, category=None, branch=None, tolerance=None):
    if df.empty:
        return df

    if tolerance is None:
        tolerance = round_window(round_no)

    lower = rank
    upper = rank + tolerance
    result = df[(df["cutoff_rank"] >= lower) & (df["cutoff_rank"] <= upper)].copy()
    if round_no is not None:
        result["round"] = pd.to_numeric(result["round"], errors="coerce")
        result = result[result["round"] == round_no]
    if category:
        category_set = normalize_category_query(category)
        result = result[result["category"].str.upper().isin(category_set)]
    if branch:
        result = result[result["branch"].apply(lambda value: branch_matches_query(value, branch))]
    if not result.empty:
        result["round"] = pd.to_numeric(result["round"], errors="coerce")
        result["cutoff_rank"] = pd.to_numeric(result["cutoff_rank"], errors="coerce")

    if not result.empty:
        result = result.sort_values(["cutoff_rank", "college_name", "branch"], ascending=[True, True, True])
    return result

def branch_college_list(df):
    if df.empty:
        return pd.DataFrame()

    branch_rows = []
    seen = set()
    for college_name, college_rows in df.groupby("college_name", sort=False):
        college_rows = college_rows.sort_values(["cutoff_rank", "branch"], ascending=[True, True])
        for _, row in college_rows.iterrows():
            round_value = int(row["round"])
            cutoff_value = row["cutoff_rank"]
            cutoff_display = int(cutoff_value) if float(cutoff_value).is_integer() else cutoff_value
            key = (
                college_name,
                row["branch"],
                round_value,
                str(row["category"]).upper(),
                cutoff_display,
                int(row["year"]),
            )
            if key in seen:
                continue
            seen.add(key)
            branch_rows.append({
                "college_name": college_name,
                "branch": row["branch"],
                "round": round_value,
                "category": str(row["category"]).upper(),
                "cutoff": cutoff_display,
                "year": int(row["year"]),
            })

    out = pd.DataFrame(branch_rows)
    if out.empty:
        return out
    return out.sort_values(["cutoff", "college_name", "branch"], ascending=[True, True, True]).reset_index(drop=True)

def rank_bucket(rank, cutoff):
    gap = float(cutoff) - float(rank)
    if gap <= 0:
        return "Reached"
    if gap <= 2000:
        return "Dream"
    if gap <= 5000:
        return "Competitive"
    return "Safe"

def bucket_summary_frame(df, rank):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "college_name": row["college_name"],
            "branch": row["branch"],
            "bucket": rank_bucket(rank, row["cutoff"]),
            "round": int(row["round"]),
            "category": row["category"],
            "cutoff": row["cutoff"],
            "year": int(row["year"]),
        })
    return pd.DataFrame(rows).sort_values(["bucket", "cutoff", "college_name"], ascending=[True, True, True]).reset_index(drop=True)

def add_years_column(df, source_df=None):
    if df is None or df.empty:
        return df
    source_df = source_df if source_df is not None else df
    year_map = {}
    grouped = source_df.groupby(["college_name", "branch"], dropna=False)
    for (college_name, branch), group in grouped:
        years = sorted({int(y) for y in group["year"].dropna().tolist() if pd.notna(y)})
        year_map[(college_name, branch)] = years
    out = df.copy()
    out["years_available"] = out.apply(
        lambda row: ", ".join(str(y) for y in year_map.get((row["college_name"], row["branch"]), [])) or "-",
        axis=1
    )
    return out

def build_data_summary(df):
    if df.empty:
        return {
            "colleges": 0,
            "branches": 0,
            "years": [],
            "rounds": [],
            "status": "No cutoff data loaded",
        }

    years = sorted({int(y) for y in df["year"].dropna().unique().tolist()})
    rounds = sorted({int(r) for r in df["round"].dropna().unique().tolist()})
    return {
        "colleges": int(df["college_name"].nunique()),
        "branches": int(df["branch"].apply(branch_family_name).nunique()),
        "years": years,
        "rounds": rounds,
        "status": "Cutoff data loaded",
    }

def sorted_unique(values):
    cleaned = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            cleaned.append(text)
    return sorted(cleaned)

def branch_matches_query(row_branch, query_branch):
    if not row_branch or not query_branch:
        return False
    row_norm = normalize_branch_query(branch_family_name(row_branch) or row_branch)
    query_norm = normalize_branch_query(branch_family_name(query_branch) or query_branch)
    if not row_norm or not query_norm:
        return False
    if row_norm == query_norm:
        return True
    alias_pairs = {
        "computer science": [
            "computer science and engineering",
            "computer science and business systems",
            "computer science and design",
            "computer science and technology",
        ],
        "information science": ["information science and engineering"],
        "electronic and communication": ["electronics and communication engineering"],
        "electronics and communication": ["electronics and communication engineering"],
        "mechanical": ["mechanical engineering"],
        "ece": ["electronics and communication engineering"],
        "ise": ["information science and engineering"],
        "cse": ["computer science and engineering"],
    }
    query_key = normalize_branch_query(query_branch)
    row_key = normalize_branch_query(row_branch)
    for alias, targets in alias_pairs.items():
        if alias in query_key and any(target in row_norm for target in targets):
            return True
    return row_norm == query_norm

def get_dropdown_options(df):
    if df.empty:
        return ["All"], ["All"], ["All"]

    round_options = sorted({int(r) for r in df["round"].dropna().tolist()})
    category_options = sorted_unique(df["category"].astype(str).str.upper().tolist())
    branch_options = sorted_unique(
        [branch_family_name(v) or v for v in df["branch"].dropna().tolist()]
    )
    return ["All"] + [str(r) for r in round_options], ["All"] + category_options, ["All"] + branch_options

def has_relevant_context(vectorstore, question, threshold=0.35):
    try:
        matches = vectorstore.similarity_search_with_relevance_scores(question, k=3)
        if not matches:
            return False, []
        top_score = matches[0][1]
        return top_score >= threshold, matches
    except Exception:
        return False, []

def build_mistral_prompt(question, context, history_messages=None, profile=None):
    history_messages = history_messages or []
    profile = profile or {}
    history_text = []
    for msg in history_messages[-6:]:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "").strip()
        if content:
            history_text.append(f"{role}: {content}")

    history_block = "\n".join(history_text) if history_text else "No prior chat history."
    profile_block = "\n".join([f"{k.capitalize()}: {v}" for k, v in profile.items()]) if profile else "No saved KCET profile."
    return f"""You are a KCET college assistant for a student.

Use the provided context and chat history as the factual source.
You may answer any user question, but if the question is about KCET colleges, branches, cutoffs, eligibility, rounds, or rank-based choices, use only the provided cutoff context.
Do not invent colleges, branches, cutoffs, categories, or policies.
If the context does not support the answer, say that clearly and ask for the missing data or relevant file.

When helpful:
- Explain rank vs cutoff in a practical way.
- Group options as Dream, Competitive, Safe, or Reached.
- Return all eligible colleges for a requested branch when the context supports it.
- Keep the answer concise, useful, and student-friendly.

Chat history:
{history_block}

Saved student profile:
{profile_block}

Retrieved context:
{context}

Question:
{question}

Answer:"""

def question_looks_like_kcet_query(question):
    if not question:
        return False
    text = question.lower()
    keywords = [
        "kcet", "cutoff", "cut-off", "rank", "college", "branch",
        "engineering", "ise", "cse", "ece", "eee", "mech", "round",
        "category", "gm", "gmr", "gmk", "stg", "scg", "2ag", "2bg",
        "3ag", "3bg"
    ]
    return any(keyword in text for keyword in keywords)

def extract_rank_from_question(question):
    if not question:
        return None
    match = re.search(r"\b(?:rank|rnk)\s*(?:is|=|:|around|about|approx\.?)?\s*(\d{1,6})\b", question, re.I)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\b\d{3,6}\b", question)
    if numbers:
        try:
            return int(numbers[0])
        except Exception:
            return None
    return None

def extract_round_from_question(question):
    if not question:
        return None
    match = re.search(r"\bround\s*[-:]?\s*(?:no\.?\s*)?([123])\b", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\br\s*[-:]?\s*([123])\b", question, re.I)
    if match:
        return int(match.group(1))
    return None

def extract_category_from_question(question, df=None):
    if not question:
        return None
    text = question.upper()
    candidates = [
        "GM", "GMR", "GMK", "SCG", "SCK", "SCR", "STG", "STK", "STR",
        "2AG", "2AK", "2AR", "2BG", "2BK", "2BR", "3AG", "3AK", "3AR",
        "3BG", "3BK", "3BR", "1G", "1K", "1R"
    ]
    for cat in candidates:
        if re.search(rf"\b{re.escape(cat)}\b", text):
            return cat
    if re.search(r"\bgeneral\s*category\b", text):
        return "GM"
    if "general" in text:
        return "GM"
    if df is not None and not df.empty:
        categories = sorted_unique(df["category"].astype(str).str.upper().tolist())
        for cat in categories:
            if re.search(rf"\b{re.escape(cat)}\b", text):
                return cat
    return None

def extract_branch_from_question(question, df=None):
    if not question:
        return None
    normalized_question = normalize_branch_query(question)
    branch_aliases = {
        "cse": "Computer Science and Engineering",
        "ise": "Information Science and Engineering",
        "ece": "Electronics and Communication Engineering",
        "eee": "Electrical and Electronics Engineering",
        "mech": "Mechanical Engineering",
        "mechanical": "Mechanical Engineering",
        "ai ml": "Artificial Intelligence and Machine Learning",
        "aiml": "Artificial Intelligence and Machine Learning",
    }
    for alias, canonical in branch_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized_question):
            return canonical
    explicit_phrases = [
        "computer science and engineering",
        "information science and engineering",
        "electronics and communication engineering",
        "electrical and electronics engineering",
        "mechanical engineering",
        "artificial intelligence and machine learning",
        "artificial intelligence and data science",
    ]
    for phrase in explicit_phrases:
        if phrase in normalized_question:
            return branch_family_name(phrase) or phrase.title()
    if df is not None and not df.empty:
        branches = canonical_branch_options(df["branch"])
        for branch in branches:
            if branch == "All":
                continue
            branch_norm = normalize_branch_query(branch)
            if branch_norm and branch_norm in normalized_question:
                return branch
    return None

def extract_college_from_question(question, df=None):
    if not question or df is None or df.empty:
        return None
    question_norm = normalize_branch_query(question)
    explicit_clues = [
        "only",
        "specific college",
        "this college",
        "for this college",
        "college wise",
        "college-specific",
    ]
    if not any(clue in question_norm for clue in explicit_clues):
        return None
    candidates = sorted_unique(df["college_name"].astype(str).tolist())
    best_match = None
    best_len = 0
    for college in candidates:
        college_norm = normalize_branch_query(college)
        if not college_norm:
            continue
        if college_norm in question_norm and len(college_norm) > best_len:
            best_match = college
            best_len = len(college_norm)
    return best_match

def missing_kcet_details(question, df=None):
    missing = []
    if extract_rank_from_question(question) is None:
        missing.append("rank")
    if extract_round_from_question(question) is None:
        missing.append("round")
    if extract_category_from_question(question, df) is None:
        missing.append("category")
    if extract_branch_from_question(question, df) is None:
        missing.append("branch")
    return missing

def ask_for_missing_details(missing):
    prompts = {
        "rank": "What is your KCET rank?",
        "round": "Which round should I use - Round 1, 2, or 3?",
        "category": "What is your category, such as GM, GMR, GMK, SCG, STG, 2AG, or 2BG?",
        "branch": "Which branch do you want, such as CSE, ISE, ECE, or Mechanical?",
    }
    ordered = [prompts[key] for key in ["rank", "round", "category", "branch"] if key in missing]
    return "Please share these details first:\n- " + "\n- ".join(ordered)

def looks_like_shortlist_request(question):
    if not question:
        return False
    text = question.lower()
    triggers = [
        "give me colleges",
        "show colleges",
        "find colleges",
        "college list",
        "shortlist",
        "based on the info",
        "based on this info",
        "based on my info",
        "based on details",
        "what colleges",
    ]
    return any(trigger in text for trigger in triggers)

def parse_kcet_profile(question):
    profile = {}
    rank = extract_rank_from_question(question)
    round_no = extract_round_from_question(question)
    category = extract_category_from_question(question)
    branch = extract_branch_from_question(question)
    if rank is not None:
        profile["rank"] = rank
    if round_no is not None:
        profile["round"] = round_no
    if category is not None:
        profile["category"] = category
    if branch is not None:
        profile["branch"] = branch
    return profile

def get_effective_kcet_profile(question, df=None):
    profile = st.session_state.get("kcet_profile", {}).copy()
    profile.update(parse_kcet_profile(question))
    st.session_state.kcet_profile = profile
    missing = []
    if "rank" not in profile:
        missing.append("rank")
    if "round" not in profile:
        missing.append("round")
    if "category" not in profile:
        missing.append("category")
    if "branch" not in profile:
        missing.append("branch")
    return profile, missing

def get_kcet_profile_missing_only(profile):
    missing = []
    if "rank" not in profile:
        missing.append("rank")
    if "round" not in profile:
        missing.append("round")
    if "category" not in profile:
        missing.append("category")
    if "branch" not in profile:
        missing.append("branch")
    return missing

def format_shortlist_answer(df, rank):
    if df is None or df.empty:
        return ""
    lines = [
        "| College | Branch | Bucket | Round | Category | Cutoff | Year | Years available |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for _, row in df.head(30).iterrows():
        lines.append(
            f"| {row['college_name']} | {row['branch']} | {row['bucket']} | "
            f"{int(row['round'])} | {row['category']} | {int(row['cutoff'])} | {int(row['year'])} | {row.get('years_available', '-')} |"
        )
    return "\n".join(lines)

def shortlist_dataframe(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["college_name", "branch", "bucket", "round", "category", "cutoff", "year", "years_available"])
    out = df.copy()
    cols = ["college_name", "branch", "bucket", "round", "category", "cutoff", "year", "years_available"]
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out[cols].reset_index(drop=True)

def trend_label(year_series):
    ordered = [x for x in year_series if pd.notna(x)]
    if len(ordered) < 2:
        return "Insufficient"
    ordered = sorted(ordered, key=lambda item: item[0])
    first_year, first_cutoff = ordered[0]
    last_year, last_cutoff = ordered[-1]
    delta = float(last_cutoff) - float(first_cutoff)
    if abs(delta) <= 500:
        direction = "Stable"
    elif delta < 0:
        direction = "Improving"
    else:
        direction = "Rising"
    return f"{direction} ({first_year}:{int(first_cutoff)} -> {last_year}:{int(last_cutoff)})"

def build_multi_year_comparison_table(df):
    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    if "cutoff" not in working.columns and "cutoff_rank" in working.columns:
        working["cutoff"] = working["cutoff_rank"]
    working["cutoff"] = pd.to_numeric(working["cutoff"], errors="coerce")
    working = working.dropna(subset=["college_name", "branch", "round", "category", "cutoff", "year"])

    group_cols = ["college_name", "branch", "round", "category"]
    year_columns = sorted({int(y) for y in working["year"].dropna().unique().tolist()})

    rows = []
    for keys, group in working.groupby(group_cols, dropna=False):
        college_name, branch, round_no, category = keys
        row = {
            "college_name": college_name,
            "branch": branch,
            "round": int(round_no),
            "category": str(category).upper(),
        }
        year_cutoffs = []
        for year in year_columns:
            year_group = group[group["year"].astype(int) == year]
            if year_group.empty:
                row[str(year)] = "-"
                continue
            best_cutoff = year_group["cutoff"].min()
            row[str(year)] = int(best_cutoff) if float(best_cutoff).is_integer() else round(float(best_cutoff), 2)
            year_cutoffs.append((year, best_cutoff))

        available_years = [str(int(y)) for y in group["year"].dropna().tolist()]
        row["years_available"] = ", ".join(sorted(set(available_years), key=lambda x: int(x))) if available_years else "-"
        row["trend"] = trend_label([(int(y), c) for y, c in year_cutoffs])
        row["best_cutoff"] = int(group["cutoff"].min()) if float(group["cutoff"].min()).is_integer() else round(float(group["cutoff"].min()), 2)
        row["bucket"] = rank_bucket(int(group["cutoff"].min()), group["cutoff"].min())
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ordered_cols = ["college_name", "branch", "bucket", "round", "category", "best_cutoff"] + [str(y) for y in year_columns] + ["trend", "years_available"]
    for col in ordered_cols:
        if col not in out.columns:
            out[col] = "-"
    return out[ordered_cols].sort_values(["bucket", "best_cutoff", "college_name"], ascending=[True, True, True]).reset_index(drop=True)

def ask_mistral(question, context, history_messages=None, profile=None):
    if not MISTRAL_API_KEY or Mistral is None:
        return None

    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = build_mistral_prompt(question, context, history_messages, profile=profile)

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful KCET admission planning assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        return response.choices[0].message.content
    except Exception:
        return None

def setup_llm():
    local_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    if GROQ_API_KEY and ChatGroq is not None:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    if ChatOllama is not None:
        return ChatOllama(model=local_model, temperature=0)

    raise RuntimeError(
        "No supported LLM backend is available. Install 'langchain-ollama' or provide GROQ_API_KEY."
    )

def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    llm = setup_llm()
    
    # Create a combined prompt template
    prompt_template = f"""{system_prompt}

{negative_prompt}

Context (from College Name documents):
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    memory = ConversationBufferMemory(
        llm = llm,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages = True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        memory = memory,
        verbose = True,
        return_source_documents = True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

def answer_question(vectorstores, conversational_chain, question, history_messages=None, cutoff_df=None):
    cutoff_df = cutoff_df if cutoff_df is not None else pd.DataFrame()
    profile = st.session_state.get("kcet_profile", {}).copy()
    profile.update(parse_kcet_profile(question))
    st.session_state.kcet_profile = profile

    context_chunks = []
    sources = []
    shortlist_df = pd.DataFrame()
    comparison_df = pd.DataFrame()
    if not cutoff_df.empty:
        rank = profile.get("rank")
        round_no = profile.get("round")
        category = profile.get("category")
        branch = profile.get("branch")
        filtered = cutoff_df.copy()
        if rank is not None:
            filtered = filtered[pd.to_numeric(filtered["cutoff_rank"], errors="coerce") >= rank]
        if round_no is not None:
            filtered = filtered[pd.to_numeric(filtered["round"], errors="coerce") == round_no]
        if category:
            filtered = filtered[filtered["category"].astype(str).str.upper().isin(normalize_category_query(category))]
        if branch:
            filtered = filtered[filtered["branch"].apply(lambda value: branch_matches_query(value, branch))]
        shortlist_df = bucket_summary_frame(branch_college_list(filtered), rank or 0)
        if not shortlist_df.empty:
            shortlist_df = add_years_column(shortlist_df, filtered)
            comparison_df = build_multi_year_comparison_table(filtered)
            shortlist_text = format_shortlist_answer(shortlist_df, rank or 0)
            context_chunks.insert(0, "KCET shortlist candidates:\n" + shortlist_text)
            sources = list(dict.fromkeys(["KCET cutoff data"] + sources))

    if not context_chunks and cutoff_df.empty:
        raw_pdf_table = st.session_state.get("cached_uploaded_pdf_table", pd.DataFrame())
        if not raw_pdf_table.empty:
            for _, row in raw_pdf_table.head(8).iterrows():
                text = str(row.get("text", "")).strip()
                source = str(row.get("source", "uploaded pdf"))
                if text:
                    context_chunks.append(f"Source: {source}\n{text[:6000]}")
                    if source not in sources:
                        sources.append(source)

    context_text = "\n\n".join(context_chunks[:8])
    history_messages = history_messages or []
    mistral_answer = ask_mistral(question, context_text, history_messages, profile=profile)
    if mistral_answer:
        return mistral_answer, sources, comparison_df if not comparison_df.empty else shortlist_dataframe(shortlist_df)
    if context_text:
        return context_text, sources, comparison_df if not comparison_df.empty else shortlist_dataframe(shortlist_df)
    return "Mistral is not configured. Please add MISTRAL_API_KEY to your .env file.", sources, comparison_df if not comparison_df.empty else shortlist_dataframe(shortlist_df)

st.set_page_config(
    page_title="Chat with College Name's Chatbot",
    page_icon="🧠",
    layout="wide",  # Changed to wide layout to accommodate sidebar
)

# Custom CSS for sidebar styling
st.markdown("""
    <style>
    div.css-textbarboxtype {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    
    /* Justify text for Purpose section */
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Upload Data")
    st.markdown("Upload cutoff files in PDF, CSV, or Excel format.")
    uploaded_files = st.file_uploader(
        "Cutoff files",
        type=["pdf", "csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help="Upload your KCET cutoff PDFs directly, or CSV/Excel if you have them.",
    )

st.title("KCET College Finder")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_base_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "kcet_profile" not in st.session_state:
    st.session_state.kcet_profile = {}
if "college_results" not in st.session_state:
    st.session_state.college_results = pd.DataFrame(columns=["college_name", "branch", "round", "category", "cutoff", "year"])

files_sig = file_signature(uploaded_files)
if st.session_state.get("cached_files_sig") != files_sig:
    st.session_state.cached_files_sig = files_sig
    st.session_state.cached_cutoff_df = load_uploaded_cutoff_data(uploaded_files)
    st.session_state.cached_uploaded_pdf_files = [f for f in (uploaded_files or []) if f.name.lower().endswith(".pdf")]
    st.session_state.cached_uploaded_pdf_vectorstore = build_uploaded_pdf_vectorstore(st.session_state.cached_uploaded_pdf_files)
    st.session_state.cached_uploaded_pdf_table = pd.DataFrame(load_uploaded_pdfs(st.session_state.cached_uploaded_pdf_files))

cutoff_df = st.session_state.get("cached_cutoff_df", pd.DataFrame())
uploaded_pdf_files = st.session_state.get("cached_uploaded_pdf_files", [])
uploaded_pdf_vectorstore = st.session_state.get("cached_uploaded_pdf_vectorstore")
uploaded_pdf_table = st.session_state.get("cached_uploaded_pdf_table", pd.DataFrame())
st.session_state.college_df = cutoff_df

summary = build_data_summary(cutoff_df)
summary_cols = st.columns(4)
summary_cols[0].metric("Colleges", summary["colleges"])
summary_cols[1].metric("Branches", summary["branches"])
summary_cols[2].metric("Rounds", ", ".join(map(str, summary["rounds"])) if summary["rounds"] else "-")
summary_cols[3].metric("Years", ", ".join(map(str, summary["years"])) if summary["years"] else "-")

st.caption(summary["status"])

st.subheader("Ask in Natural Language")
st.caption("Tell me your rank, category, round, and branch. I’ll ask for any missing details before showing the shortlist.")

round_options, category_options, branch_options = get_dropdown_options(cutoff_df)

with st.form("kcet_intake_form", clear_on_submit=False):
    form_col1, form_col2, form_col3, form_col4 = st.columns(4)
    with form_col1:
        form_rank = st.text_input("Rank", value=str(st.session_state.kcet_profile.get("rank", "")))
    with form_col2:
        current_round = str(st.session_state.kcet_profile.get("round", ""))
        round_index = round_options.index(current_round) if current_round in round_options else 0
        form_round = st.selectbox("Round", round_options, index=round_index)
    with form_col3:
        current_category = str(st.session_state.kcet_profile.get("category", ""))
        category_index = category_options.index(current_category) if current_category in category_options else 0
        form_category = st.selectbox("Category", category_options, index=category_index)
    with form_col4:
        current_branch = str(st.session_state.kcet_profile.get("branch", ""))
        branch_index = branch_options.index(current_branch) if current_branch in branch_options else 0
        form_branch = st.selectbox("Branch", branch_options, index=branch_index)
    submitted = st.form_submit_button("Save details")

if submitted:
    if form_rank.strip().isdigit():
        st.session_state.kcet_profile["rank"] = int(form_rank.strip())
    if str(form_round).isdigit():
        st.session_state.kcet_profile["round"] = int(str(form_round))
    if form_category and form_category != "All":
        st.session_state.kcet_profile["category"] = str(form_category).upper()
    else:
        st.session_state.kcet_profile.pop("category", None)
    if form_branch and form_branch != "All":
        st.session_state.kcet_profile["branch"] = str(form_branch)
    else:
        st.session_state.kcet_profile.pop("branch", None)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": "Saved your details. You can now ask for colleges or paste one more detail if needed."}
    )

if st.button("Reset KCET details"):
    st.session_state.kcet_profile = {}
    st.session_state.chat_history = []
    st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Example: I have 18000 rank, GM category, Round 3, CSE")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.kcet_profile.update(parse_kcet_profile(user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        assistant_response, sources, comparison_df = answer_question(
            [st.session_state.vectorstore, uploaded_pdf_vectorstore],
            st.session_state.conversational_chain,
            user_input,
            history_messages=st.session_state.chat_history,
            cutoff_df=cutoff_df
        )
        if comparison_df is not None and not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(assistant_response)
        if sources:
            st.caption("Sources: " + ", ".join(sources))
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
