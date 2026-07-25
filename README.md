# Bajío AI Solutions — Agente RAG 🤖

Este proyecto es la entrega final para el **Challenge Alura Agente**. Consiste en un asistente virtual inteligente impulsado por Inteligencia Artificial (modelo RAG) diseñado para consultar y responder dudas operativas basándose estrictamente en la Base de Conocimiento oficial de **Bajío AI Solutions**.

## 🏗️ Arquitectura de la Solución

El agente utiliza la arquitectura **RAG (Retrieval-Augmented Generation)** para asegurar que las respuestas sean precisas y limitadas al contexto de la empresa, evitando alucinaciones del modelo.

1. **Extracción:** Carga de la Base de Conocimiento mediante `PyPDF2`.
2. **Chunking:** División del texto en fragmentos manejables utilizando `RecursiveCharacterTextSplitter` de LangChain.
3. **Embeddings:** Vectorización de los fragmentos con `gemini-embedding-001`.
4. **Almacenamiento:** Indexación vectorial en memoria utilizando `FAISS`.
5. **Generación:** Orquestación del prompt y generación de la respuesta final mediante `Google Gemini 2.0 Flash`.

## 🛠️ Tecnologías y Herramientas

* **Lenguaje:** Python 3.11+
* **Interfaz Gráfica:** Streamlit
* **Orquestación RAG:** LangChain
* **LLM & Embeddings:** Google Generative AI (Gemini)
* **Base de Datos Vectorial:** FAISS (Facebook AI Similarity Search)

## 🚀 Instrucciones de Ejecución Local

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/tu-usuario/bajio-ai-rag.git](https://github.com/tu-usuario/bajio-ai-rag.git)
   cd bajio-ai-rag
