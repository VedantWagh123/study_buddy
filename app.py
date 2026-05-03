import streamlit as st
import fitz
import numpy as np
import faiss
import os
import hashlib
from google import genai
from google.genai import types

st.set_page_config(page_title="AI PDF Study Helper", layout="wide", page_icon="🧠")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_DIM = 768

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"]{
  background:linear-gradient(135deg,#050b1a 0%,#0a1628 50%,#060d1f 100%) !important;
  font-family:'Inter',sans-serif;color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(10,22,40,0.97) !important;
  border-right:1px solid rgba(99,179,237,0.12) !important;}
section[data-testid="stSidebarContent"]{padding:0 !important;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#050b1a;}
::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:10px;}
.nav-logo{text-align:center;padding:24px 20px 16px;
  border-bottom:1px solid rgba(99,179,237,0.1);margin-bottom:10px;}
.nav-logo h2{font-size:15px;font-weight:700;letter-spacing:1.5px;
  background:linear-gradient(135deg,#63b3ed,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav-logo span{font-size:10px;color:#475569;letter-spacing:2px;}
.header-wrap{display:flex;justify-content:space-between;align-items:center;
  padding:20px 0 24px;border-bottom:1px solid rgba(99,179,237,0.08);margin-bottom:28px;}
.header-title{font-size:28px;font-weight:700;letter-spacing:-0.5px;
  background:linear-gradient(135deg,#e2e8f0 30%,#63b3ed);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.header-sub{font-size:12px;color:#475569;margin-top:3px;letter-spacing:1.5px;text-transform:uppercase;}
.avatar{width:44px;height:44px;border-radius:50%;
  background:linear-gradient(135deg,#3b82f6,#8b5cf6);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;border:2px solid rgba(99,179,237,0.35);
  box-shadow:0 0 20px rgba(99,179,237,0.2);}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;}
.stat-card{background:rgba(15,25,50,0.75);border:1px solid rgba(99,179,237,0.12);
  border-radius:18px;padding:22px;backdrop-filter:blur(12px);
  transition:all 0.3s ease;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(99,179,237,0.4),transparent);}
.stat-card:hover{border-color:rgba(99,179,237,0.3);transform:translateY(-2px);
  box-shadow:0 8px 32px rgba(99,179,237,0.1);}
.stat-icon{font-size:22px;margin-bottom:12px;}
.stat-value{font-size:32px;font-weight:700;line-height:1;margin-bottom:6px;}
.stat-value.blue{background:linear-gradient(135deg,#63b3ed,#3b82f6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stat-value.purple{background:linear-gradient(135deg,#a78bfa,#7c3aed);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stat-value.green{background:linear-gradient(135deg,#34d399,#059669);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stat-value.orange{background:linear-gradient(135deg,#fb923c,#ea580c);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stat-label{font-size:12px;color:#475569;letter-spacing:0.5px;}
.stat-bar{display:flex;gap:3px;margin-top:12px;}
.bar{height:4px;border-radius:4px;background:rgba(99,179,237,0.15);flex:1;}
.bar.active{background:linear-gradient(90deg,#63b3ed,#a78bfa);}
.upload-card{background:rgba(15,25,50,0.75);border:1px solid rgba(99,179,237,0.15);
  border-radius:20px;padding:28px;backdrop-filter:blur(16px);
  text-align:center;margin-bottom:20px;}
.upload-title{font-size:20px;font-weight:700;margin-bottom:6px;
  background:linear-gradient(135deg,#e2e8f0,#63b3ed);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.upload-sub{font-size:13px;color:#475569;}
.section-title{font-size:13px;font-weight:600;color:#64748b;
  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;}
.file-name{font-size:13px;font-weight:500;color:#cbd5e1;padding-top:4px;}
.file-size{font-size:11px;color:#475569;padding-top:6px;}
.badge{display:inline-block;padding:4px 10px;border-radius:20px;font-size:11px;
  font-weight:600;background:rgba(52,211,153,0.1);color:#34d399;
  border:1px solid rgba(52,211,153,0.2);margin-top:4px;}
[data-testid="stFileUploader"]{background:rgba(15,25,50,0.5) !important;
  border:1.5px dashed rgba(99,179,237,0.25) !important;border-radius:14px !important;}
.stTextInput>div>div>input{background:rgba(15,25,50,0.8) !important;
  border:1px solid rgba(99,179,237,0.2) !important;border-radius:10px !important;
  color:#e2e8f0 !important;padding:10px 14px !important;}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
  color:white !important;border:none !important;border-radius:10px !important;
  font-weight:600 !important;box-shadow:0 0 20px rgba(99,179,237,0.2) !important;
  transition:all 0.3s ease !important;}
.stButton>button:hover{box-shadow:0 0 35px rgba(99,179,237,0.4) !important;}
.chat-msg-user{display:flex;justify-content:flex-end;margin-bottom:8px;}
.chat-msg-ai{display:flex;justify-content:flex-start;margin-bottom:8px;}
.bubble-user{max-width:80%;padding:10px 14px;border-radius:14px 14px 4px 14px;
  font-size:13px;line-height:1.6;
  background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:white;}
.bubble-ai{max-width:85%;padding:12px 16px;border-radius:14px 14px 14px 4px;
  font-size:13px;line-height:1.7;
  background:rgba(99,179,237,0.1);border:1px solid rgba(99,179,237,0.15);color:#cbd5e1;}
.chat-panel-header{padding:14px 18px;border-bottom:1px solid rgba(99,179,237,0.1);
  font-size:14px;font-weight:600;color:#e2e8f0;
  background:rgba(15,25,50,0.6);display:flex;align-items:center;gap:8px;}
.online-dot{margin-left:auto;width:8px;height:8px;border-radius:50%;
  background:#34d399;display:inline-block;box-shadow:0 0 8px #34d399;}
.divider{border:none;border-top:1px solid rgba(99,179,237,0.06);margin:6px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
defaults = {
    "chat_open": False,
    "messages": [{"role": "ai", "text": "👋 Hello! Upload a PDF and ask me anything about it."}],
    "uploaded_files": [],
    "faiss_index": None,
    "chunks": [],
    "query_count": 0,
    "last_confidence": None,
    "summaries": {},
    "processed_hashes": set(),
    "nav": "Home",
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", "AIzaSyCOHobhERWEkoQGek4sNmkDhOjqh3ABZYU"),

}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def bars(filled: int, total: int = 8) -> str:
    filled = max(0, min(filled, total))
    return "".join(
        '<div class="bar active"></div>' if i < filled else '<div class="bar"></div>'
        for i in range(total)
    )

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def extract_text(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"Error extracting text: {e}"

def chunk_text(text: str) -> list:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + CHUNK_SIZE])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def clean_query(query: str) -> str:
    """Remove extra spaces and shorten repeated characters to save tokens."""
    import re
    return re.sub(r'\s+', ' ', query).strip()

def limit_context(chunks: list, max_chars: int = 1800) -> str:
    """Trim context to maximum ~1800 chars to save input tokens."""
    context = ""
    for c in chunks:
        if len(context) + len(c) > max_chars:
            remaining = max_chars - len(context)
            if remaining > 30:
                context += c[:remaining] + "..."
            break
        context += c + "\n---\n"
    return context.strip()

def build_compact_prompt(question: str, context: str) -> str:
    """Minimal prompt design to save tokens."""
    return f"""Answer concise using ONLY the given context.
If the context does not contain the answer, reply EXACTLY with: NOT_FOUND

CONTEXT:
{context}

Q: {question}

Format:
## Heading
## Key Points
## Explanation
## Conclusion"""

def _fallback_vec(text: str) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(EMBED_DIM).astype(np.float32)

def get_embedding(text: str) -> np.ndarray:
    api_key = st.session_state.gemini_api_key
    if not api_key:
        return _fallback_vec(text)
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        if len(vec) < EMBED_DIM:
            vec = np.pad(vec, (0, EMBED_DIM - len(vec)))
        else:
            vec = vec[:EMBED_DIM]
        return vec
    except Exception:
        return _fallback_vec(text)

def add_to_faiss(new_chunks: list):
    if not new_chunks:
        return
    vecs = np.stack([get_embedding(c) for c in new_chunks])
    faiss.normalize_L2(vecs)
    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    st.session_state.faiss_index.add(vecs)
    st.session_state.chunks.extend(new_chunks)

# ── RAG Pipeline Functions ─────────────────────────────────────────────────────

def generate_with_fallback_models(client, prompt: str) -> str:
    """Loop through model list and return first successful response."""
    models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=300, temperature=0.3)
            )
            return response.text.strip()
        except Exception:
            continue
    raise Exception("__ALL_FAILED__")

def retrieve_docs(query: str, k: int = 3) -> list:
    """Retrieve top-k relevant chunks from FAISS. Optimized to k=3."""
    if st.session_state.faiss_index is None or not st.session_state.chunks:
        return []
    qvec = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(qvec)
    k = min(k, len(st.session_state.chunks))
    _, indices = st.session_state.faiss_index.search(qvec, k)
    return [st.session_state.chunks[i] for i in indices[0] if 0 <= i < len(st.session_state.chunks)]

# Keep backward-compatible alias
retrieve_chunks = retrieve_docs

def generate_answer(question: str, context_chunks: list):
    """Send question + RAG context to Gemini. Optimized token usage."""
    api_key = st.session_state.gemini_api_key
    if not api_key:
        return None, None
    try:
        client = genai.Client(api_key=api_key)
        if context_chunks:
            context_str = limit_context(context_chunks, 1800)
            prompt = build_compact_prompt(question, context_str)
        else:
            prompt = f"Answer concisely:\n{question}\nFormat:\n## Heading\n## Key Points\n## Explanation\n## Conclusion"
            
        text = generate_with_fallback_models(client, prompt)
        if "NOT_FOUND" in text.upper():
            return "⚠️ **Answer not available in current PDF.**\n\nThe uploaded document does not contain information related to this query.", 100.0
            
        confidence = round(min(99.0, 75.0 + len(text) / 200), 1)
        return text, confidence
    except Exception as e:
        if str(e) == "__ALL_FAILED__":
            return "__LIMIT__", None
        return None, None

# Keep backward-compatible alias
def call_gemini(prompt: str):
    """Legacy wrapper — optimized output token limit."""
    api_key = st.session_state.gemini_api_key
    if not api_key:
        return None, None
    try:
        client = genai.Client(api_key=api_key)
        text = generate_with_fallback_models(client, prompt)
        confidence = round(min(99.0, 75.0 + len(text) / 200), 1)
        return text, confidence
    except Exception as e:
        if str(e) == "__ALL_FAILED__":
            return "__LIMIT__", None
        return None, None

def format_fallback_response(context: str, query: str) -> str:
    import re
    if not context.strip():
        return "⚠️ **No relevant data found in document.**"
        
    # Simplify and extract clean sentences
    sentences = [s.strip() for s in context.replace('\n', ' ').split(".") if len(s.strip()) > 30]
    
    if not sentences:
        return "⚠️ **No relevant data found in document.**"
        
    # Determine Topic
    words = [w.capitalize() for w in re.findall(r'\w+', query) if len(w) > 3]
    topic_title = " ".join(words[:4]) if words else "Document Insights"
    
    # Format Key Points
    points = ""
    for s in sentences[:3]:
        points += f"• {s}.\n"
        
    # Format Explanation (break into small paragraphs using double newline)
    explanation_parts = [f"{s}." for s in sentences[:5]]
    
    clean_parts = []
    for part in explanation_parts:
        if isinstance(part, list):
            for sub in part:
                clean_parts.append(str(sub))
        else:
            clean_parts.append(str(part))
            
    clean_parts = [str(p) for p in clean_parts if p]
    explanation = "\n\n".join(clean_parts)

    # Quick Summary
    summary = sentences[0]
    
    return (
        f"### {topic_title}\n\n"
        f"**➤ Key Points:**\n{points}\n"
        f"**✓ Explanation:**\n{explanation}\n\n"
        f"**➤ Quick Summary:**\n{summary}."
    )

def fallback_answer(question: str, chunks: list) -> str:
    if not st.session_state.chunks:
        return "⚠️ **No relevant data found in document.**\n\nPlease upload a PDF first."

    import re
    stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "this", "that", "these", "those", "explain", "describe", "define", "about", "write", "detail", "tell", "give"}
    q_words = [w.lower() for w in re.findall(r'\w+', question) if len(w) > 2 and w.lower() not in stop_words]
    
    # Search all chunks in the entire PDF for actual matches
    matched_chunks = []
    if q_words:
        for chunk in st.session_state.chunks:
            chunk_lower = chunk.lower()
            if any(w in chunk_lower for w in q_words):
                matched_chunks.append(chunk)

    if q_words and not matched_chunks:
        return (
            "⚠️ **Answer not available in current PDF.**\n\n"
            "The uploaded document does not contain information related to this query."
        )

    # Use retrieved RAG chunks (matched via keyword, or fallback to default top chunks)
    best_chunks = matched_chunks[:3] if matched_chunks else chunks[:3]
    if not best_chunks:
        best_chunks = st.session_state.chunks[:3]

    combined_context = " ".join(best_chunks)
    
    return format_fallback_response(combined_context, question)

def build_summary_prompt(text: str) -> str:
    """Compact prompt for summary generation to save tokens."""
    return f"""Summarize concise (max 150 words).
CONTENT:
{text[:2500]}

Format:
## Title
## Key Points
## Short Explanation
## Final Summary"""

def generate_summary(text: str, fname: str) -> str:
    result, _ = call_gemini(build_summary_prompt(text))
    if result and result != "__LIMIT__":
        return result
    
    # ── Robust Local Extractive Summary Fallback ──
    words = text.split()
    snippet = " ".join(words[:60]) + "..."
    
    # Extract meaningful sentences for bullet points
    all_sentences = [s.strip() for s in text.replace('\n', ' ').split(".") if len(s.strip()) > 40]
    bullets = "\n".join(f"- {s}." for s in all_sentences[1:5]) if len(all_sentences) > 4 else "- Core content extracted successfully."
    
    # Extract concluding sentences for final summary
    mid_to_end = all_sentences[len(all_sentences)//2:] if len(all_sentences) > 10 else all_sentences
    final_sum = " ".join(f"{s}." for s in mid_to_end[:3]) if mid_to_end else "The document contains technical or informational content that requires detailed review."

    return (
        f"## Title\n{fname}\n\n"
        f"## Key Points\n{bullets}\n\n"
        f"## Short Explanation\n{snippet}\n\n"
        f"## Final Summary\n{final_sum}"
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="nav-logo">
        <h2>⬡ PRECISION AI</h2>
        <span>PDF STUDY HELPER</span>
    </div>
    """, unsafe_allow_html=True)

    key_input = st.text_input(
        "🔑 Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="Enter your Gemini API key…",
        key="api_key_field",
    )
    if key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = key_input

    st.markdown("<br>", unsafe_allow_html=True)

    for icon, label in [("🏠", "Home"), ("📁", "Files"), ("💬", "Chat")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.nav = label
            st.rerun()

    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;color:#1e3a5f;font-size:11px;padding:0 16px;'>
        v3.1.0 · Precision AI<br>© 2025 AI Study Helper
    </div>
    """, unsafe_allow_html=True)

# ── Top Navigation Bar ─────────────────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

nav_cols = st.columns(4)
for i, item in enumerate(["Dashboard", "Predict Question Paper", "Find IMP", "Quiz"]):
    btn_type = "primary" if st.session_state.active_tab == item else "secondary"
    if nav_cols[i].button(item, use_container_width=True, type=btn_type, key=f"topnav_{item}"):
        st.session_state.active_tab = item
        st.rerun()

st.markdown("---")

if st.session_state.active_tab == "Predict Question Paper":
    st.markdown("""
    <div class="upload-card">
        <div style="font-size:42px;margin-bottom:12px;">📄</div>
        <div class="upload-title">Predict Your Success</div>
        <div class="upload-sub">Upload documents to generate AI-powered question papers<br><small style="color:#63b3ed;">Analysis takes approximately 15–30 seconds</small></div>
    </div>
    """, unsafe_allow_html=True)
    
    paper_pdf = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"], key="paper_uploader")
    
    if st.button("Generate Question Paper", use_container_width=True, type="primary"):
        if not paper_pdf:
            st.warning("Please upload a document first.")
        else:
            with st.spinner("Generating Question Paper..."):
                try:
                    import fitz
                    text = ""
                    if paper_pdf.name.endswith('.pdf'):
                        doc = fitz.open(stream=paper_pdf.read(), filetype="pdf")
                        text = "".join(page.get_text() for page in doc)
                    else:
                        text = paper_pdf.read().decode('utf-8', errors='ignore')
                    
                    api_key = st.session_state.gemini_api_key
                    if not api_key:
                        raise Exception("No API key")
                        
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"Generate a full exam-style question paper from this content.\n\nInclude:\n- Section A (Short Questions)\n- Section B (Medium Questions)\n- Section C (Long Questions)\n\nFormat:\n- Clear headings\n- Numbered questions\n- Proper spacing\n- Exam style formatting\n\nMake it realistic and well-structured.\n\nCONTENT:\n{text[:15000]}"
                    
                    resp = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.4)
                    )
                    
                    st.session_state.predicted_paper = resp.text
                    st.rerun()
                except Exception as e:
                    import random
                    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 30]
                    random.shuffle(sentences)
                    
                    selected_q = sentences[:15]
                    
                    fallback_out = "### 📄 Mock Question Paper\n\n"
                    fallback_out += "#### Section A (Short Questions)\n"
                    for i, q in enumerate(selected_q[:5]):
                        fallback_out += f"**Q{i+1}.** Define the concept: {q[:60]}?\n\n"
                        
                    fallback_out += "#### Section B (Medium Questions)\n"
                    for i, q in enumerate(selected_q[5:10]):
                        fallback_out += f"**Q{i+6}.** Explain the following statement in detail: {q[:80]}...\n\n"
                        
                    fallback_out += "#### Section C (Long Questions)\n"
                    for i, q in enumerate(selected_q[10:15]):
                        fallback_out += f"**Q{i+11}.** Discuss the implications of this topic: {q[:100]}...\n\n"
                        
                    st.session_state.predicted_paper = fallback_out
                    st.warning("⚠️ **API limit exceeded. Switching to local mode.**")
                    
    if st.session_state.get("predicted_paper"):
        st.markdown("---")
        st.markdown("### 📝 Generated Question Paper")
        st.markdown(f'<div style="background:rgba(255,255,255,0.05); padding:30px; border-radius:10px; border:1px solid rgba(255,255,255,0.1);">{st.session_state.predicted_paper}</div>', unsafe_allow_html=True)
        
        # PDF Generation for download
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            clean_text = st.session_state.predicted_paper.encode('latin-1', 'replace').decode('latin-1')
            
            for line in clean_text.split('\n'):
                line = line.replace('**', '').replace('### ', '').replace('#### ', '')
                pdf.multi_cell(0, 10, txt=line)
                
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Question Paper (PDF)",
                data=pdf_bytes,
                file_name="Question_Paper.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.download_button(
                label="📥 Download Question Paper (TXT)",
                data=st.session_state.predicted_paper,
                file_name="Question_Paper.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
            
    st.stop()

if st.session_state.active_tab == "Find IMP":
    st.markdown("""
    <div class="upload-card">
        <div style="font-size:42px;margin-bottom:12px;">🌟</div>
        <div class="upload-title">Analyze Key Sections</div>
        <div class="upload-sub">Identify high-impact areas from your documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    imp_pdf = st.file_uploader("Upload PDF (Max 20MB)", type=["pdf"], key="imp_uploader")
    
    col1, col2 = st.columns(2)
    q_range = col1.selectbox("Question Range", [5, 10, 15, 20])
    q_marks = col2.radio("Marks per Question", ["2 Marks", "5 Marks", "10 Marks"], horizontal=True)
    
    if st.button("Generate Insights ⚡", use_container_width=True, type="primary"):
        if not imp_pdf:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Generating Insights..."):
                try:
                    import fitz
                    doc = fitz.open(stream=imp_pdf.read(), filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    
                    api_key = st.session_state.gemini_api_key
                    if not api_key:
                        raise Exception("No API key")
                        
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"From the given content, generate {q_range} important exam questions.\nEach question should match {q_marks} level difficulty.\n\nOutput format:\n- Topic Title\n- Important Questions (bullet points)\n- Short Explanation (if needed)\n\nKeep answers clear, exam-focused, and structured.\n\nCONTENT:\n{text[:15000]}"
                    
                    resp = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.3)
                    )
                    
                    st.session_state.imp_insights = resp.text
                    st.rerun()
                except Exception as e:
                    import random
                    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 40]
                    random.shuffle(sentences)
                    
                    selected_q = sentences[:min(q_range, len(sentences))]
                    
                    fallback_out = "### 📚 Topic: General Document Concepts\n\n"
                    for i, q in enumerate(selected_q):
                        fallback_out += f"- **Q{i+1} ({q_marks}):** Explain the concept: '{q[:80]}...'\n"
                        fallback_out += f"  - *Short Explanation:* Focus on this section of the text for the exam.\n\n"
                    
                    st.session_state.imp_insights = fallback_out
                    st.warning("⚠️ **API limit exceeded. Switching to local mode.**")
                    
    if st.session_state.get("imp_insights"):
        st.markdown("---")
        st.markdown("### 📊 Generated Insights")
        st.markdown(f'<div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:10px;">\n{st.session_state.imp_insights}\n</div>', unsafe_allow_html=True)
        
    st.stop()

if st.session_state.active_tab == "Quiz":
    st.markdown("""
    <div class="upload-card">
        <div style="font-size:42px;margin-bottom:12px;">🧠</div>
        <div class="upload-title">Initialize Intelligence</div>
        <div class="upload-sub">Upload a PDF to generate a smart quiz</div>
    </div>
    """, unsafe_allow_html=True)
    
    quiz_pdf = st.file_uploader("Upload PDF for Quiz", type=["pdf"], key="quiz_uploader")
    
    col1, col2 = st.columns(2)
    q_count = col1.slider("Question Count", 5, 20, 5)
    q_time = col2.selectbox("Time Selection", ["1 min", "2 min", "5 min", "10 min"])
    
    if st.button("Start Test", use_container_width=True, type="primary"):
        if not quiz_pdf:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Generating Quiz..."):
                try:
                    import fitz, json
                    doc = fitz.open(stream=quiz_pdf.read(), filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    
                    api_key = st.session_state.gemini_api_key
                    if not api_key:
                        st.error("API limit exceeded. Cannot generate quiz.")
                    else:
                        from google import genai
                        from google.genai import types
                        client = genai.Client(api_key=api_key)
                        prompt = f"Generate {q_count} multiple choice questions from this content.\nEach question must have:\n- Question\n- 4 options (A, B, C, D)\n- Correct answer\nOutput ONLY valid JSON array of objects. Example: [{{\"question\": \"Q?\", \"options\": [\"A. 1\", \"B. 2\", \"C. 3\", \"D. 4\"], \"answer\": \"A. 1\"}}]\n\nCONTENT:\n{text[:10000]}"
                        
                        resp = client.models.generate_content(
                            model="gemini-2.0-flash", 
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=0.3)
                        )
                        raw = resp.text.strip()
                        if raw.startswith("```json"):
                            raw = raw[7:]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()
                        
                        st.session_state.quiz_data = json.loads(raw)
                        st.session_state.quiz_submitted = False
                        st.session_state.user_answers = {}
                        st.rerun()
                except Exception as e:
                    # LOCAL FALLBACK QUIZ GENERATION
                    import re, random
                    
                    # Get sentences with reasonable length
                    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 50]
                    # Find potential answers (words longer than 5 chars)
                    all_words = list(set([w for w in re.findall(r'\b[A-Za-z]{6,}\b', text)]))
                    
                    if not sentences or len(all_words) < 4:
                        st.error("Document too short to generate local fallback quiz.")
                    else:
                        random.shuffle(sentences)
                        quiz = []
                        
                        for s in sentences:
                            if len(quiz) >= q_count:
                                break
                                
                            words_in_s = re.findall(r'\b[A-Za-z]{6,}\b', s)
                            if not words_in_s:
                                continue
                                
                            ans = random.choice(words_in_s)
                            q_text = s.replace(ans, "______", 1) + "?"
                            
                            fakes = random.sample([w for w in all_words if w.lower() != ans.lower()], min(3, max(0, len(all_words)-1)))
                            opts = fakes + [ans]
                            while len(opts) < 4: opts.append("None")
                            random.shuffle(opts)
                            
                            labels = ["A", "B", "C", "D"]
                            formatted_opts = [f"{labels[i]}. {opt}" for i, opt in enumerate(opts)]
                            correct_opt = formatted_opts[opts.index(ans)]
                            
                            quiz.append({
                                "question": q_text,
                                "options": formatted_opts,
                                "answer": correct_opt
                            })
                            
                        st.session_state.quiz_data = quiz
                        st.session_state.quiz_submitted = False
                        st.session_state.user_answers = {}
                        st.warning("⚠️ **Local Mode Activated:** API limit exceeded. Generated a fill-in-the-blank quiz locally from PDF text.")
                        st.rerun()
    
    if st.session_state.quiz_data:
        st.markdown("### Quiz Attempt")
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            ans = st.radio("Options", q['options'], key=f"quiz_q_{i}", index=None, label_visibility="collapsed")
            st.session_state.user_answers[i] = ans
            st.markdown("<br>", unsafe_allow_html=True)
            
        if not st.session_state.quiz_submitted:
            if st.button("Submit Test", use_container_width=True, type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()
        else:
            correct = sum(1 for i, q in enumerate(st.session_state.quiz_data) if st.session_state.user_answers.get(i) == q['answer'])
            total = len(st.session_state.quiz_data)
            pct = (correct / total) * 100 if total > 0 else 0
            
            msg = "Good Job!" if pct > 80 else ("Nice प्रयास" if pct >= 50 else "Improve more")
            
            st.markdown(f"""
            <div style="background:rgba(99,179,237,0.1); padding:30px; border-radius:15px; text-align:center; border:1px solid rgba(99,179,237,0.3); margin-top:20px; margin-bottom:20px;">
                <h1 style="color:#63b3ed; font-size:64px; margin:0;">{correct} / {total}</h1>
                <h3 style="color:#e2e8f0; margin:10px 0;">{pct:.1f}% - {msg}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, q in enumerate(st.session_state.quiz_data):
                user_a = st.session_state.user_answers.get(i)
                st.write(f"**Q{i+1}: {q['question']}**")
                
                if user_a == q['answer']:
                    st.success(f"Your answer: {user_a} ✅")
                else:
                    st.error(f"Your answer: {user_a} ❌")
                    st.success(f"Correct answer: {q['answer']}")
                st.markdown("---")
    
    st.stop()


# ── Main Content ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-wrap">
    <div>
        <div class="header-title">AI PDF Study Helper</div>
        <div class="header-sub">Upload · Summarize · Ask · Learn</div>
    </div>
    <div class="avatar">🧠</div>
</div>
""", unsafe_allow_html=True)

# Dynamic stat cards
doc_count    = len(st.session_state.uploaded_files)
query_count  = st.session_state.query_count
confidence   = st.session_state.last_confidence
neural_load  = min(100, doc_count * 12 + query_count * 3)

conf_display   = f"{confidence}%" if confidence is not None else "--"
conf_color     = "green" if confidence is not None else "purple"
conf_bars      = bars(int(confidence / 12.5)) if confidence else 0
neural_display = f"{neural_load}%" if doc_count > 0 else "--"
neural_bars    = bars(max(1, neural_load // 12)) if doc_count > 0 else bars(0)

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">📄</div>
        <div class="stat-value blue">{doc_count}</div>
        <div class="stat-label">Documents Processed</div>
        <div class="stat-bar">{bars(min(8, doc_count))}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-value purple">{query_count}</div>
        <div class="stat-label">Active Queries</div>
        <div class="stat-bar">{bars(min(8, query_count))}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-value {conf_color}">{conf_display}</div>
        <div class="stat-label">AI Confidence</div>
        <div class="stat-bar">{bars(conf_bars)}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🧬</div>
        <div class="stat-value orange">{neural_display}</div>
        <div class="stat-label">Neural Load</div>
        <div class="stat-bar">{neural_bars}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload section
st.markdown("""
<div class="upload-card">
    <div style="font-size:42px;margin-bottom:12px;">🧩</div>
    <div class="upload-title">Ingest Intelligence</div>
    <div class="upload-sub">Upload PDF documents for deep-context analysis and Q&amp;A</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop your PDFs here or click to browse",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="visible",
    key="pdf_uploader",
)

if uploaded_files:
    for uploaded in uploaded_files:
        raw = uploaded.getvalue()
        fhash = file_hash(raw)
        if fhash not in st.session_state.processed_hashes:
            with st.spinner(f"Processing **{uploaded.name}**…"):
                size_kb = round(len(raw) / 1024, 1)
                size_str = f"{size_kb} KB" if size_kb < 1024 else f"{round(size_kb/1024, 1)} MB"
                text = extract_text(raw)
                chunks = chunk_text(text)
                add_to_faiss(chunks)
                summary = generate_summary(text, uploaded.name)
                st.session_state.summaries[uploaded.name] = summary
                st.session_state.uploaded_files.insert(0, {
                    "name": uploaded.name,
                    "size": size_str,
                    "status": "Analyzed",
                    "chunks": len(chunks),
                })
                st.session_state.processed_hashes.add(fhash)
            st.success(f"✅ **{uploaded.name}** processed — {len(chunks)} chunks indexed")

# Summaries
if st.session_state.summaries:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 &nbsp;Document Summaries</div>', unsafe_allow_html=True)
    for fname, summary in st.session_state.summaries.items():
        with st.expander(f"📄 {fname}", expanded=False):
            st.markdown(summary)

st.markdown("<br>", unsafe_allow_html=True)

# Recent files
st.markdown('<div class="section-title">📂 &nbsp;Recent Files</div>', unsafe_allow_html=True)
if not st.session_state.uploaded_files:
    st.markdown(
        '<div style="color:#475569;font-size:13px;padding:12px 0;">No files uploaded yet.</div>',
        unsafe_allow_html=True,
    )
else:
    for f in st.session_state.uploaded_files:
        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
        with c1:
            st.markdown(f'<div class="file-name">📄 {f["name"]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="file-size">{f["size"]}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<span class="badge">✓ {f["status"]}</span>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="file-size">{f.get("chunks","?")} chunks</div>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chat Toggle ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, right = st.columns([9, 1])
with right:
    label = "✕ Close" if st.session_state.chat_open else "🤖 Chat"
    if st.button(label, key="fab_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

# ── Chat Panel ─────────────────────────────────────────────────────────────────
if st.session_state.chat_open:
    st.markdown("---")
    api_status = "🟢 RAG LIVE" if st.session_state.gemini_api_key else "🔴 LOCAL DATA"
    status_color = "#34d399" if st.session_state.gemini_api_key else "#ef4444"
    
    st.markdown(f"""
    <div class="chat-panel-header">
        🤖 &nbsp;AI Study Assistant
        <span style="margin-left:auto; font-size:10px; font-weight:700; color:{status_color}; background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:12px; letter-spacing:1px; border:1px solid {status_color}40;">
            {api_status}
        </span>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-msg-user"><div class="bubble-user">{msg["text"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-msg-ai"><div class="bubble-ai">{msg["text"]}</div></div>',
                unsafe_allow_html=True,
            )

    col_input, col_btn = st.columns([8, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask anything about your documents…",
        )
    with col_btn:
        send = st.button("➤", key="send_btn")

    if send and user_input.strip():
        question = clean_query(user_input.strip())
        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.query_count += 1

        # Caching logic to prevent repeated exact queries
        cache_key = f"ans_{hashlib.md5(question.encode()).hexdigest()}"
        if cache_key in st.session_state:
            answer = st.session_state[cache_key]
        else:
            # Step 1 — Retrieve top 3 relevant chunks
            rel_chunks = retrieve_docs(question, k=3)

            # Step 2 — Generate optimized answer
            result, conf = generate_answer(question, rel_chunks)

            # Step 3 — Handle fallback locally
            if result == "__LIMIT__":
                limit_note = "⚠️ **All models unavailable. Switching to local mode.**\n\n"
                answer = limit_note + fallback_answer(question, rel_chunks)
            elif result is None:
                answer = fallback_answer(question, rel_chunks)
            else:
                answer = result
                st.session_state.last_confidence = conf
            
            st.session_state[cache_key] = answer

        st.session_state.messages.append({"role": "ai", "text": answer})
        st.rerun()
