"""
Bajío AI Solutions — Agente RAG Inteligente
Responde preguntas estrictamente basadas en la Base de Conocimiento oficial.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────
load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "Bajio_AI_Solutions_Base_De_Conocimiento.pdf")

SYSTEM_PROMPT = """
Eres el asistente virtual oficial de **Bajío AI Solutions**.
Tu única fuente de información es el contexto proporcionado a continuación, extraído de la
Base de Conocimiento interna de la empresa.

**Reglas estrictas:**
1. Responde ÚNICAMENTE con información presente en el contexto.
2. Si la pregunta no puede responderse con el contexto, di:
   "Lo siento, no cuento con esa información en mi base de conocimiento actual.
    Te recomiendo contactar al equipo de soporte de Bajío AI Solutions."
3. Responde siempre en español, de forma profesional, clara y concisa.
4. Cuando sea pertinente, estructura tu respuesta con viñetas o numeración.

Contexto:
{context}

Pregunta del usuario:
{question}

Respuesta:
"""


# ──────────────────────────────────────────────
# 1. PDF Loading & Text Extraction
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 Leyendo el PDF de la Base de Conocimiento…")
def load_pdf_text(path: str) -> str:
    """Extract all text from a local PDF file."""
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text


# ──────────────────────────────────────────────
# 2. Text Chunking
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="✂️ Dividiendo el texto en fragmentos…")
def split_text_into_chunks(_text: str) -> list:
    """Split raw text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents([_text])


# ──────────────────────────────────────────────
# 3. Embedding + FAISS Vector Store
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Generando embeddings y construyendo el índice FAISS…")
def build_vector_store(_chunks) -> FAISS:
    """Create a FAISS index from document chunks using Google embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return FAISS.from_documents(_chunks, embedding=embeddings)


# ──────────────────────────────────────────────
# 4. QA Chain (Gemini LLM)
# ──────────────────────────────────────────────
def get_qa_chain():
    """Build a LangChain QA chain using Google Gemini."""
    from langchain.chains.question_answering import load_qa_chain
    prompt = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)
# ──────────────────────────────────────────────
# 5. Streamlit UI
# ──────────────────────────────────────────────
def main():
    # ── Page config ──
    st.set_page_config(
        page_title="Bajío AI Solutions — Asistente RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #1a1a40 40%, #24243e 100%);
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.95);
            border-right: 1px solid rgba(99, 102, 241, 0.15);
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li {
            color: #c7c7e0;
        }

        /* ── Header ── */
        .hero-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .hero-sub {
            text-align: center;
            color: #94a3b8;
            font-size: 1.05rem;
            margin-top: 4px;
            margin-bottom: 28px;
        }

        /* ── Chat messages ── */
        .stChatMessage {
            border-radius: 16px !important;
            padding: 14px 18px !important;
            backdrop-filter: blur(12px);
        }
        [data-testid="stChatMessageContent"] p {
            font-size: 0.97rem;
            line-height: 1.65;
        }

        /* ── Glassmorphism card ── */
        .glass-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(99, 102, 241, 0.12);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(14px);
            margin-bottom: 20px;
        }

        /* ── Input ── */
        .stChatInput > div {
            border: 1px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 14px !important;
            background: rgba(255,255,255,0.03) !important;
        }
        .stChatInput > div:focus-within {
            border-color: #818cf8 !important;
            box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.25) !important;
        }

        /* ── Status badges ── */
        .status-ready {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-waiting {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(250, 204, 21, 0.12);
            color: #facc15;
            border: 1px solid rgba(250, 204, 21, 0.3);
        }

        /* ── Footer ── */
        .footer-text {
            text-align: center;
            color: #475569;
            font-size: 0.78rem;
            margin-top: 48px;
            padding-bottom: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🔑 Configuración")

        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            placeholder="Pega tu clave aquí…",
            help="Obtén tu clave en https://aistudio.google.com/app/apikey",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input

        st.markdown("---")
        st.markdown("### 📖 Acerca del agente")
        st.markdown(
            """
            Este asistente utiliza **RAG** (Retrieval-Augmented Generation)
            para responder preguntas basándose **exclusivamente** en la
            Base de Conocimiento oficial de **Bajío AI Solutions**.

            **Stack tecnológico:**
            - 🧠 Google Gemini 2.0 Flash
            - 📦 FAISS (vector search)
            - 🔗 LangChain
            - 🖥️ Streamlit
            """
        )
        st.markdown("---")
        st.markdown(
            """
            **Arquitectura del pipeline:**
            1. Extracción de texto (PyPDF2)
            2. Chunking (RecursiveTextSplitter)
            3. Embeddings (Google Embedding-001)
            4. Búsqueda vectorial (FAISS)
            5. Generación (Gemini 2.0 Flash)
            """
        )

    # ── Hero ──
    st.markdown('<p class="hero-title">🤖 Bajío AI Solutions</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Asistente Inteligente RAG — Base de Conocimiento Interna</p>', unsafe_allow_html=True)

    # ── API key guard ──
    active_key = st.session_state.get("api_key") or os.getenv("GOOGLE_API_KEY")
    if not active_key:
        st.markdown(
            '<div class="glass-card">'
            '<span class="status-waiting">⏳ Esperando API Key</span>'
            '<p style="color:#94a3b8; margin-top:12px;">'
            'Introduce tu <strong>Google API Key</strong> en la barra lateral para activar el agente.'
            '</p></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.session_state["api_key"] = active_key

    # ── Load pipeline ──
    raw_text = load_pdf_text(PDF_PATH)
    chunks = split_text_into_chunks(raw_text)
    vector_store = build_vector_store(chunks)
    qa_chain = get_qa_chain()

    # ── Status ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="glass-card" style="text-align:center">'
            f'<div style="font-size:1.8rem;font-weight:700;color:#818cf8">{len(chunks)}</div>'
            '<div style="color:#94a3b8;font-size:0.85rem">Fragmentos indexados</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="glass-card" style="text-align:center">'
            '<div style="font-size:1.8rem;font-weight:700;color:#a78bfa">FAISS</div>'
            '<div style="color:#94a3b8;font-size:0.85rem">Motor de búsqueda vectorial</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="glass-card" style="text-align:center">'
            '<span class="status-ready">✅ Agente listo</span>'
            '<div style="color:#94a3b8;font-size:0.85rem;margin-top:6px">Gemini 2.0 Flash</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Chat history ──
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! 👋 Soy el asistente virtual de **Bajío AI Solutions**.\n\n"
                    "Puedo responder preguntas sobre:\n"
                    "- 🛠️ Nuestro portafolio de servicios\n"
                    "- ⏱️ Tiempos de implementación\n"
                    "- ☁️ Infraestructura y despliegue en la nube\n"
                    "- 🔗 Integraciones con sistemas existentes\n"
                    "- 🔒 Políticas de privacidad y soporte\n\n"
                    "¿En qué puedo ayudarte?"
                ),
            }
        ]

    # ── Render messages ──
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── User input ──
    if user_query := st.chat_input("Escribe tu pregunta aquí…"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        # Retrieve & generate
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Buscando en la base de conocimiento…"):
                relevant_docs = vector_store.similarity_search(user_query, k=4)
                response = qa_chain.invoke(
                    {"input_documents": relevant_docs, "question": user_query}
                )
                answer = response["output_text"]

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── Footer ──
    st.markdown(
        '<p class="footer-text">'
        '© 2026 Bajío AI Solutions · Agente RAG con Google Gemini · Documento interno'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
