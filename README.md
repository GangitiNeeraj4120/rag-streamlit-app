# 📚 Chat with Multiple PDFs — RAG Application

Ever wanted to ask questions directly from a PDF instead of scrolling through dozens of pages?

This project is a simple **Retrieval-Augmented Generation (RAG)** application that lets you upload one or more PDFs and chat with them using natural language.

The app extracts the text from your PDFs, breaks it into smaller chunks, converts those chunks into embeddings, and stores them in a **FAISS vector database**. When you ask a question, the application finds the most relevant pieces of information from your documents and gives them to **Llama 3.1 8B Instant via Groq** to generate an answer.

### 🌐 Try the App

**[👉 Open the live Streamlit app](https://rag-app-app-ckgqs8qnjtp8tsni9vbi7x.streamlit.app/)**

> Upload your own PDFs and ask questions about them directly in the browser.

---

## 🚀 What Can It Do?

* 📄 Upload multiple PDF files at once
* 🔍 Extract text from the uploaded documents
* ✂️ Split large documents into smaller, overlapping chunks
* 🧠 Convert text into semantic embeddings using Hugging Face
* 🗃️ Store embeddings in a FAISS vector database
* 🔎 Retrieve the most relevant document chunks for a question
* 🤖 Use Llama 3.1 8B Instant through Groq to generate answers
* 💬 Ask questions about the uploaded documents in natural language
* 🖥️ Use everything through a simple Streamlit interface

---

## 🧠 Why RAG?

A normal LLM doesn't automatically know the contents of a PDF that you just uploaded.

Instead of sending the entire PDF to the LLM every time, this project uses a **RAG pipeline**:

1. The PDF is converted into text.
2. The text is split into manageable chunks.
3. Each chunk is converted into a vector embedding.
4. The embeddings are stored in FAISS.
5. When a question is asked, FAISS finds the most relevant chunks.
6. Those chunks are added to the prompt as context.
7. Llama generates the final answer using that context.

In simple terms:

**PDF → Retrieve relevant information → Give it to the LLM → Generate answer**

This helps the model focus on the information contained in the uploaded documents rather than trying to answer purely from its pretrained knowledge.

---

## 🏗️ How the Application Works

```text
                  Upload PDFs
                      │
                      ▼
              ┌─────────────────┐
              │   PyPDFLoader   │
              │  Extract Text   │
              └────────┬────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Text Chunking        │
            │ Chunk Size: 1000     │
            │ Overlap: 200         │
            └──────────┬───────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Hugging Face       │
             │ MiniLM Embeddings  │
             └──────────┬─────────┘
                        │
                        ▼
                ┌──────────────┐
                │    FAISS     │
                │ Vector Store │
                └──────┬───────┘
                       │
                 User Question
                       │
                       ▼
              Similarity Search
                       │
                       ▼
              Relevant Chunks
                       │
                       ▼
              ┌────────────────┐
              │  Llama 3.1 8B  │
              │  via Groq      │
              └───────┬────────┘
                      │
                      ▼
                 Final Answer
```

---

## 🔄 Project Workflow

### 1. Upload the PDFs

The application uses Streamlit's file uploader to allow multiple PDFs to be uploaded at the same time.

### 2. Extract the Text

`PyPDFLoader` reads each PDF and extracts its text along with useful document metadata such as page information.

### 3. Split the Text

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

The current configuration is:

| Parameter     | Value |
| ------------- | ----: |
| Chunk size    |  1000 |
| Chunk overlap |   200 |

The overlap helps retain context when a relevant piece of information happens to cross chunk boundaries.

### 4. Create Embeddings

Each chunk is converted into a vector using:

`sentence-transformers/all-MiniLM-L6-v2`

These vectors allow the application to compare the meaning of the user's question with the meaning of the document chunks.

### 5. Store and Retrieve with FAISS

The embeddings are stored in **FAISS**.

When a question is asked, FAISS performs a similarity search to find the document chunks that are most relevant to the question.

### 6. Generate the Answer

The retrieved chunks are added to a prompt along with the user's question.

The prompt instructs the model to answer using the provided context rather than relying on unrelated information.

The response is generated using:

**Llama 3.1 8B Instant → Groq**

---

## 🛠️ Tech Stack

| Technology               | Why I Used It                           |
| ------------------------ | --------------------------------------- |
| **Python**               | Main programming language               |
| **Streamlit**            | Build the interactive web UI            |
| **LangChain**            | Connect the different RAG components    |
| **PyPDF**                | Extract text from PDFs                  |
| **Hugging Face**         | Generate text embeddings                |
| **FAISS**                | Store and search vector embeddings      |
| **Groq**                 | Fast LLM inference                      |
| **Llama 3.1 8B Instant** | Generate answers from retrieved context |
| **python-dotenv**        | Manage API keys securely                |

---

## 💻 Running It Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd pdf-rag-chatbot
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Make sure `.env` is included in `.gitignore` so your API key isn't accidentally pushed to GitHub.

### 5. Start the application

```bash
streamlit run app.py
```

The application should then be available at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
pdf-rag-chatbot/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 📦 Main Dependencies

```text
langchain==0.3.26
langchain-core==0.3.66
langchain-community==0.3.26
langchain-text-splitters==0.3.8
langchain-groq
sentence-transformers
faiss-cpu
pypdf
streamlit
python-dotenv
tf-keras
```

---

## ⚠️ Current Limitations

This is a learning/project implementation, so there are a few things I'd improve in a production version.

### Vector store is session-based

The FAISS vector store currently lives in Streamlit's session state. If the application restarts, the PDFs need to be uploaded and processed again.

### No source citations yet

The application retrieves document chunks internally, but the final answer doesn't currently display the source PDF and page number.

Adding source citations would make the answers easier to verify.

### Limited retrieval configuration

The current similarity search uses the default retrieval settings. A future version could allow control over the number of retrieved chunks and add a reranking stage.

### Scanned PDFs

PDFs that contain scanned images rather than selectable text may require OCR before they can be processed effectively.

---

## 🔮 What I'd Like to Improve Next

Some improvements I'm considering for the project:

* 💾 Persist the FAISS index between sessions
* 📌 Show PDF names and page numbers as answer sources
* 🔄 Improve handling when users add or remove PDFs
* 💬 Add conversational chat history
* 🔢 Make the number of retrieved chunks configurable
* 🧠 Experiment with reranking models
* 📄 Add support for DOCX and TXT files
* 📊 Add RAG evaluation metrics
* ☁️ Improve deployment and scalability

---

## 🎯 What I Learned

Building this project helped me understand how a RAG application works beyond simply calling an LLM API.

Some of the main concepts I worked with are:

* Retrieval-Augmented Generation
* Document loading and preprocessing
* Text chunking
* Vector embeddings
* Semantic similarity search
* FAISS vector databases
* Prompt engineering
* LLM integration
* LangChain
* Streamlit
* API-based model inference

The main idea I wanted to understand was how **external knowledge can be retrieved from documents and provided to an LLM at query time**.

---

## 👨‍💻 Author

**Neeraj**

A hands-on project exploring **Generative AI, LLM applications, and Retrieval-Augmented Generation**.

### 🌐 Live Demo

**[Try the PDF RAG App on Streamlit →](https://rag-app-app-ckgqs8qnjtp8tsni9vbi7x.streamlit.app/)**

If you find the project useful, feel free to ⭐ the repository!
