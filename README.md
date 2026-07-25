
# Bajío AI Solutions — Agente RAG 🤖

Este proyecto es la entrega final para el **Challenge Alura Agente**. Consiste en un asistente virtual inteligente impulsado por Inteligencia Artificial (modelo RAG) diseñado para consultar y responder dudas operativas basándose estrictamente en la Base de Conocimiento oficial de **Bajío AI Solutions**.

---

## 🌐 Despliegue en la Nube (App en Vivo)

El proyecto se encuentra desplegado y completamente funcional. Puedes probar el agente directamente en el siguiente enlace:

👉 **[Abrir Asistente Bajío AI Solutions en Render](https://bajio-ai-rag.onrender.com/)**

> **⚠️ NOTA IMPORTANTE PARA EVALUADORES (Error 429):**
> Si al interactuar con el chat aparece un error rojo indicando cuota excedida (`429 You exceeded your current quota`), significa que la API Key por defecto ha alcanzado su límite de peticiones gratuitas.
> 
> **Solución:** La aplicación fue construida con una funcionalidad específica para solucionar esto. Simplemente ve al menú lateral izquierdo, pega tu propia **Google API Key** en el campo de "Configuración" y presiona Enter. ¡Con esto podrás probar el asistente sin ningún límite!

---

## 🏗️ Arquitectura de la Solución

El agente utiliza la arquitectura **RAG (Retrieval-Augmented Generation)** para asegurar que las respuestas sean precisas y limitadas al contexto de la empresa, evitando alucinaciones del modelo.

1. **Extracción:** Carga de la Base de Conocimiento mediante `PyPDF2`.
2. **Chunking:** División del texto en fragmentos manejables utilizando `RecursiveCharacterTextSplitter` de LangChain.
3. **Embeddings:** Vectorización de los fragmentos con `models/gemini-embedding-001`.
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
git clone [https://github.com/cruz-torres-dev/bajio-ai-rag.git](https://github.com/cruz-torres-dev/bajio-ai-rag.git)
cd bajio-ai-rag

```

2. Crea y activa un entorno virtual:

```bash
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

```

3. Instala las dependencias:

```bash
pip install -r requirements.txt

```

4. Crea un archivo `.env` en la raíz del proyecto y agrega tu API Key de Google:

```env
GOOGLE_API_KEY="tu_clave_api_aqui"

```

5. Ejecuta la aplicación:

```bash
streamlit run app.py

```

```

```
