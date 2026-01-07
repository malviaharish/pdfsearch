import os
import sqlite3
import numpy as np
import streamlit as st

# ---------------- ENV DETECTION ---------------- #

IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_RUNTIME") == "cloud"

# ---------------- SAFE IMPORTS ---------------- #

try:
    import fitz  # PyMuPDF
except Exception as e:
    st.error("PyMuPDF (fitz) not available. Check pymupdf installation.")
    st.stop()

# OCR imports only if local
if not IS_STREAMLIT_CLOUD:
    try:
        import pytesseract
        from pdf2image import convert_from_path
        OCR_AVAILABLE = True
    except Exception:
        OCR_AVAILABLE = False
else:
    OCR_AVAILABLE = False

from sentence_transformers import SentenceTransformer
import faiss

# ---------------- CONFIG ---------------- #

DB = "pdf_index.db"
HIGHLIGHT_DIR = "highlighted"
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

st.set_page_config(
    page_title="Local PDF Search Engine",
    layout="wide"
)

# ---------------- MODELS ---------------- #

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

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
    if not text.strip():
        return
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pdf_index VALUES (?, ?, ?)",
        (file, page, text)
    )
    conn.commit()
    conn.close()

# ---------------- OCR ---------------- #

def ocr_pdf(pdf_path):
    pages = []
    images = convert_from_path(pdf_path)
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        pages.append((i + 1, text))
    return pages

# ---------------- PDF PROCESSING ---------------- #

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    # Digital PDF
    if any(page.get_text().strip() for page in doc):
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]

    # Scanned PDF
    if OCR_AVAILABLE:
        return ocr_pdf(pdf_path)

    # Scanned but OCR unavailable
    st.warning(f"OCR disabled for scanned PDF: {os.path.basename(pdf_path)}")
    return []

# ---------------- SEARCH ---------------- #

def boolean_match(text, query):
    text = text.lower()
    q = query.lower()

    if " and " in q:
        a, b = q.split(" and ", 1)
        return a in text and b in text
    if " or " in q:
        a, b = q.split(" or ", 1)
        return a in text or b in text
    if " not " in q:
        a, b = q.split(" not ", 1)
        return a in text and b not in text

    return q in text

def keyword_search(query):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT file, page, content FROM pdf_index")
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
    _, idx = index.search(np.array(q_emb), top_k)

    return [(rows[i][0], rows[i][1]) for i in idx[0]]

# ---------------- HIGHLIGHT ---------------- #

def highlight_pdf(pdf_path, term):
    doc = fitz.open(pdf_path)
    for page in doc:
        for rect in page.search_for(term):
            page.add_highlight_annot(rect)

    out = os.path.join(HIGHLIGHT_DIR, os.path.basename(pdf_path))
    doc.save(out)
    return out

# ---------------- UI ---------------- #

st.title("📄 PDF Search Engine (Streamlit-Safe)")

if IS_STREAMLIT_CLOUD:
    st.info("Running on Streamlit Cloud → OCR disabled")
else:
    st.success("Running locally → OCR enabled")

init_db()

# ---------- Upload ---------- #

st.header("📤 Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.button("Index PDFs"):
    with st.spinner("Indexing PDFs..."):
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())

            for page, text in process_pdf(file.name):
                store_page(file.name, page, text)

    st.success("Indexing completed")

# ---------- Search ---------- #

st.header("🔍 Search")

query = st.text_input("Enter phrase / Boolean query (AND / OR / NOT)")

col1, col2 = st.columns(2)

if col1.button("Keyword / Boolean Search"):
    st.session_state.results = keyword_search(query)

if col2.button("Semantic Search"):
    st.session_state.results = semantic_search(query)

# ---------- Results ---------- #

st.header("📊 Results")

if "results" in st.session_state:
    if st.session_state.results:
        st.table([
            {"File": f, "Page": p}
            for f, p in st.session_state.results
        ])
    else:
        st.warning("No results found")

# ---------- Highlight ---------- #

st.header("🖍️ Highlight")

if "results" in st.session_state and st.session_state.results:
    selected_file = st.selectbox(
        "Select PDF",
        sorted(set(f for f, _ in st.session_state.results))
    )

    if st.button("Highlight Term"):
        out = highlight_pdf(selected_file, query)
        with open(out, "rb") as f:
            st.download_button(
                "⬇️ Download Highlighted PDF",
                f,
                file_name=os.path.basename(out),
                mime="application/pdf"
            )
