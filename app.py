import os
import sqlite3
import fitz
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# ---------------- CONFIG ---------------- #

DB = "pdf_index.db"
HIGHLIGHT_DIR = "highlighted"
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- DATABASE ---------------- #

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pdf_index (
            file TEXT,
            page INTEGER,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_page(file, page, text):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO pdf_index VALUES (?, ?, ?)",
                (file, page, text))
    conn.commit()
    conn.close()

# ---------------- OCR ---------------- #

def ocr_pdf(pdf_path):
    text_pages = []
    images = convert_from_path(pdf_path)
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        text_pages.append((i + 1, text))
    return text_pages

# ---------------- PDF PROCESSING ---------------- #

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    if all(not page.get_text().strip() for page in doc):
        return ocr_pdf(pdf_path)

    pages = []
    for i, page in enumerate(doc):
        pages.append((i + 1, page.get_text()))
    return pages

# ---------------- SEARCH ---------------- #

def boolean_match(text, query):
    text = text.lower()
    q = query.lower()

    if " and " in q:
        a, b = q.split(" and ")
        return a in text and b in text
    if " or " in q:
        a, b = q.split(" or ")
        return a in text or b in text
    if " not " in q:
        a, b = q.split(" not ")
        return a in text and b not in text

    return q in text

def keyword_search(query):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM pdf_index")
    rows = cur.fetchall()
    conn.close()

    return [(f, p) for f, p, c in rows if boolean_match(c, query)]

# ---------------- SEMANTIC SEARCH ---------------- #

def semantic_search(query, top_k=5):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT file, page, content FROM pdf_index")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    texts = [r[2] for r in rows]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    q_emb = model.encode([query])
    _, indices = index.search(np.array(q_emb), top_k)

    return [(rows[i][0], rows[i][1]) for i in indices[0]]

# ---------------- HIGHLIGHT ---------------- #

def highlight_pdf(pdf_path, term):
    doc = fitz.open(pdf_path)
    for page in doc:
        for rect in page.search_for(term):
            page.add_highlight_annot(rect)

    out = os.path.join(HIGHLIGHT_DIR, os.path.basename(pdf_path))
    doc.save(out)
    return out

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Local PDF Search Engine", layout="wide")
st.title("📄 Local PDF Search Engine")

init_db()

# -------- Upload Section -------- #

st.header("📤 Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.button("Index PDFs"):
    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            temp_path = file.name
            with open(temp_path, "wb") as f:
                f.write(file.read())

            pages = process_pdf(temp_path)
            for page, text in pages:
                store_page(temp_path, page, text)

    st.success("PDFs indexed successfully")

# -------- Search Section -------- #

st.header("🔍 Search")
query = st.text_input("Enter phrase / Boolean query (AND / OR / NOT)")

col1, col2 = st.columns(2)

with col1:
    if st.button("Keyword / Boolean Search"):
        results = keyword_search(query)
        st.session_state["results"] = results

with col2:
    if st.button("Semantic Search"):
        results = semantic_search(query)
        st.session_state["results"] = results

# -------- Results Section -------- #

st.header("📊 Results")

if "results" in st.session_state:
    if st.session_state["results"]:
        st.table(
            [{"File": f, "Page": p} for f, p in st.session_state["results"]]
        )
    else:
        st.warning("No results found")

# -------- Highlight Section -------- #

st.header("🖍️ Highlight PDF")

if "results" in st.session_state and st.session_state["results"]:
    selected = st.selectbox(
        "Select file to highlight",
        list(set(f for f, _ in st.session_state["results"]))
    )

    if st.button("Highlight Term"):
        out = highlight_pdf(selected, query)
        with open(out, "rb") as f:
            st.download_button(
                "⬇️ Download Highlighted PDF",
                f,
                file_name=os.path.basename(out),
                mime="application/pdf"
            )
