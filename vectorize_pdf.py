import pickle
import numpy as np
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import textwrap

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 200:  # Only keep chunks with substantial content
            chunks.append(chunk)
    return chunks

def vectorize_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    """Convert text chunks to vectors using sentence-transformers."""
    model = SentenceTransformer(model_name)
    vectors = model.encode(chunks)
    return vectors

def save_vectors(vectors, chunks, output_file="vectors.pkl"):
    """Save vectors and their source chunks to a binary file."""
    data = {
        "vectors": vectors,
        "chunks": chunks,
        "metadata": {
            "model": "all-MiniLM-L6-v2",
            "dimensions": vectors.shape[1],
            "chunk_count": len(chunks)
        }
    }
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    return output_file

def view_vector_sample(vectors, n=1):
    """Display a sample vector to understand its structure."""
    sample = vectors[0]
    return sample

def vectorize_pdf(pdf_path="doc.pdf"):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Vectorize chunks
    vectors = vectorize_chunks(chunks)
    
    # Save vectors
    output_file = save_vectors(vectors, chunks)
    return f"Vectors saved ({len(chunks)} chunks)"

if __name__ == "__main__":
    vectorize_pdf()