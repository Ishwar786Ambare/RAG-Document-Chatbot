# RAG Document Chatbot - Project Overview

## 1. Project Description
This project is a powerful Document Chatbot built using FastAPI, LangChain, ChromaDB, and Google Gemini. It utilizes a technique called **RAG (Retrieval-Augmented Generation)**. The main goal of the project is to allow users to upload PDF documents and ask questions based solely on the data inside those documents. Instead of relying on pre-trained general knowledge, the AI retrieves the most relevant information from your uploaded file to generate accurate, context-aware responses.

## 2. Setup and Run Commands
To run the server locally, you can use the following commands.

### Prerequisites:
Ensure you have a `.env` file created in your root directory containing your Google API Key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Installation:
If you haven't installed the dependencies yet:
```bash
pip install -r requirements.txt
```

### Starting the Server:
Run the FastAPI application from the terminal:
```bash
uvicorn app.main:app --reload
```
Once the server is running, you can test the endpoints using the interactive Swagger UI by visiting: **http://localhost:8000/docs**

---

## 3. Flow of Document Upload
When a PDF file is submitted to the `POST /upload-pdf` endpoint, the data goes through several crucial steps to become searchable:

1. **File Storage**: The incoming PDF file is securely saved to a local folder named `uploads/`.
2. **Document Loading**: LangChain's `PyPDFLoader` is triggered to read and extract the raw text content from the saved PDF file.
3. **Text Splitter (Chunking)**: Sending a massive document to an AI model is inefficient and expensive. Instead, the `RecursiveCharacterTextSplitter` breaks the text down into smaller, digestible "chunks" (500 characters per chunk, with a 50-character overlap to preserve sentences/context shared across dividing lines).
4. **Creating Embeddings**: The chunks of text are passed into a fast, open-source embedding model from HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`). This process translates the human-readable text blocks into high-dimensional numerical vectors, allowing the computer to understand the semantic "meaning" of the words.
5. **Vector Database**: These numerical vectors are then saved persistently into a local **ChromaDB** vector database inside the `chroma_db/` folder. The system updates a global `vector_store` object to let the chatbot know the document is ready to be queried.

---

## 4. Flow of Asking Questions (Getting Relevant Data)
Once the document is uploaded locally in the vector store, users can ask questions using the `POST /ask` endpoint. The flowchart works like this:

1. **Safety Check**: The endpoint first ensures that the document vectors exist. If the memory `vector_store` is empty, it prompts the user to upload a PDF first.
2. **Vector Retrieval**: When a question is asked, the system converts the question itself into a numerical vector. It scours the `ChromaDB` database and performs a *similarity search*. It fetches only the **top 3 most relevant chunks** (`k=3`) of text from the PDF that are most mathematically similar to the question being asked.
3. **Prompt Construction**: A specific prompt template is created that instructs the AI: *"You are a helpful assistant. Use the following context to answer questions."*
4. **Injecting Context**: The exact sentences/chunks retrieved from the PDF memory in step 2 are merged and injected into the prompt as the `{context}`, right alongside the user's specific `{input}` (question).
5. **AI Generation (Gemini)**: The fully packaged prompt is sent over to the **Google Gemini 2.5 Flash** language model. Because Gemini is provided with both the question AND the exact subset of document data containing the answer, it synthesizes a highly accurate, tailored response.
6. **Returning the Answer**: The generated text is parsed out and returned to the client as a clean, direct answer to their question.
