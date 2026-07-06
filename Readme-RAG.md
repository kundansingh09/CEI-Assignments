# Secure PDF RAG Assistant

## Overview
This project implements a robust, locally hosted Retrieval-Augmented Generation (RAG) system. It allows users to query custom PDF documents securely. By extracting text, generating semantic vector embeddings, and utilizing a local vector database, the system provides accurate, contextually grounded answers to user questions while preventing AI hallucinations.

This implementation emphasizes system stability, featuring built-in protections against common document parsing errors and API limitations.

## Key Features
* **Safe Semantic Segmentation:** Utilizes a hardened overlapping chunking algorithm that strictly prevents memory leaks and infinite loops caused by improper parameter configurations.
* **Ghost Document Detection:** Automatically detects and halts execution if an image-based or scanned PDF (containing no extractable text) is supplied, preventing silent failures.
* **API Rate-Limit Protection:** Incorporates controlled execution delays during the vectorization phase to ensure compliance with free-tier API rate limits (HTTP 429 errors).
* **Persistent Local Storage:** Employs ChromaDB to save document embeddings locally (`./local_vector_store`), eliminating the need to re-process large documents across sessions.
* **Strict Contextual Grounding:** Prompts are engineered to force the LLM to answer *strictly* from the provided document context.

## Prerequisites
Ensure you have Python 3.9+ installed on your system.

You will also need an active API key from Google AI Studio to access the Gemini embedding and generation models.

## Installation & Setup

**1. Clone the Repository**
Navigate to your preferred directory and initialize the project environment.

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

**3. Install Required Dependencies**

```bash
pip install chromadb pypdf google-genai python-dotenv

```

**4. Configure Environment Variables**
Create a `.env` file in the root directory of the project and add your Google API key:

```env
GEMINI_API_KEY="your_api_key_here"

```

## Configuration

Before running the application, ensure you configure the core parameters located at the top of the script (or in Cell 1 if using a Jupyter Notebook):

* `TARGET_DOCUMENT`: The exact path to the PDF file you wish to query (e.g., `"data/research_paper.pdf"`).
* `MAX_CHARS_PER_SEGMENT`: The maximum length of each text chunk (default: `600`).
* `SEGMENT_OVERLAP`: The number of characters that overlap between chunks to preserve sentence context (default: `150`). *Note: This must remain strictly less than the maximum characters.*

## Usage

If you are running this as a standard Python script, simply execute it from your terminal:

```bash
python main.py

```

**Execution Flow:**

1. **Parsing:** The system will read the PDF, check for valid text, and segment it.
2. **Vectorization:** Text segments are sent to the embedding model and safely stored in the `local_vector_store` directory.
3. **Chat Engine:** The terminal will display `--- RAG Assistant Online ---`. You can now type your queries. Type `quit` or `exit` to terminate the session.

## Troubleshooting

* **`ValueError: Configuration Error`**: Ensure `SEGMENT_OVERLAP` is smaller than `MAX_CHARS_PER_SEGMENT`.
* **`ValueError: Extraction Failed`**: The PDF you provided is likely a scanned image. You must use a text-searchable PDF, or run an OCR (Optical Character Recognition) tool on the PDF first.
* **API Errors during embedding**: Check your internet connection, verify your API key in the `.env` file, or ensure you haven't exceeded your Google AI Studio quota.

```

```