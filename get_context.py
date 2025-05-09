import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_vector_db(vector_file="vectors.pkl"):
    """Load vector database from pickle file."""
    try:
        with open(vector_file, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def clean_chunk_text(text):
    """Clean and format text chunks for better readability."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF formatting issues
    text = re.sub(r'(\w)\s+([.,:])', r'\1\2', text)  # Fix spaced punctuation
    
    # Identify and format bullet points and numbered lists
    text = re.sub(r'(\s*[•○●]\s*)', '\n• ', text)
    text = re.sub(r'(\n\d+\.)\s+', r'\1 ', text)
    
    # Identify section headers (numbered or with keywords like "Phase", "Step", etc.)
    section_header_pattern = r'(\d+\.\s*[A-Z][a-zA-Z\s]+:|(?:Phase|Step|Section|Chapter)\s+\d+:)'
    text = re.sub(section_header_pattern, r'\n\n\1', text)
    
    # Identify subsections that use letters or numbers with parentheses
    subsection_pattern = r'([a-z]\)|[0-9]\))'
    text = re.sub(subsection_pattern, r'\n\1', text)
    
    # Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure spacing after periods for better sentence separation
    text = re.sub(r'\.(\S)', r'. \1', text)
    
    # Fix dangling lines (short lines that should be part of previous paragraph)
    lines = text.split('\n')
    result_lines = []
    i = 0
    while i < len(lines):
        if i < len(lines) - 1 and len(lines[i]) < 50 and not lines[i].endswith(('.', ':', '?', '!')) and not lines[i+1].startswith(('•', '-', '*')):
            result_lines.append(lines[i] + ' ' + lines[i+1])
            i += 2
        else:
            result_lines.append(lines[i])
            i += 1
    
    text = '\n'.join(result_lines)
    
    return text.strip()

def vectorize_query(query, model_name="all-MiniLM-L6-v2"):
    """Convert query text to vector using the same model as the document chunks."""
    model = SentenceTransformer(model_name)
    query_vector = model.encode([query])
    return query_vector

def find_relevant_chunks(query_vector, vector_db, top_k=3):
    """Find the most relevant chunks using cosine similarity."""
    # Get document vectors from the database
    doc_vectors = vector_db["vectors"]
    
    # Calculate cosine similarity between query vector and all document vectors
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    # Get indices of top-k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Get the similarities scores and chunks for the top matches
    results = []
    for idx in top_indices:
        results.append({
            "chunk": vector_db["chunks"][idx],
            "similarity": similarities[idx],
            "index": idx
        })
    
    return results

def format_results(results):
    """Format the retrieved chunks into a single context string with improved readability."""
    formatted_chunks = []
    
    for r in results:
        # Clean up the chunk text
        chunk_text = r['chunk']
        
        # Replace multiple spaces with a single space
        chunk_text = ' '.join(chunk_text.split())
        
        # Replace common PDF artifacts
        chunk_text = chunk_text.replace('○', '•')
        chunk_text = chunk_text.replace('●', '•')
        
        # Convert bullet points into proper formatting
        lines = chunk_text.split('•')
        if len(lines) > 1:
            cleaned_lines = []
            for i, line in enumerate(lines):
                if i == 0 and not line.strip():
                    continue
                if i > 0:
                    cleaned_lines.append(f"• {line.strip()}")
                else:
                    cleaned_lines.append(line.strip())
            chunk_text = '\n'.join(cleaned_lines)
        
        # Format section headers
        if ':' in chunk_text and len(chunk_text.split(':')[0].split()) <= 5:
            parts = chunk_text.split(':', 1)
            if len(parts) == 2:
                chunk_text = f"{parts[0].strip()}:\n{parts[1].strip()}"
        
        # Add to formatted chunks
        formatted_chunks.append(f"[Similarity: {r['similarity']:.4f}]\n\n{chunk_text}")
    
    # Join with clear separators
    context = "\n\n" + "="*50 + "\n\n".join(formatted_chunks) + "\n\n" + "="*50
    return context

def query_vector_db(query, vector_file="vectors.pkl", top_k=3):
    """Process a query and return relevant context from the vector database."""
    # Load the vector database
    vector_db = load_vector_db(vector_file)
    if vector_db is None:
        return "Failed to load vector database."
    
    # Vectorize the query
    query_vector = vectorize_query(query, vector_db["metadata"]["model"])
    
    # Find relevant chunks
    results = find_relevant_chunks(query_vector, vector_db, top_k)
    
    # Add post-processing to clean up chunks before formatting
    for r in results:
        # Remove excessive newlines and spaces
        r['chunk'] = clean_chunk_text(r['chunk'])
    
    # Format results into context
    context = format_results(results)
    
    return context

def main():
    parser = argparse.ArgumentParser(description="Query a vector database and retrieve relevant context.")
    parser.add_argument("query", type=str, help="The query or question to search for")
    parser.add_argument("--db", type=str, default="vectors.pkl", help="Path to vector database file")
    parser.add_argument("--top", type=int, default=3, help="Number of top results to return")
    parser.add_argument("--output", type=str, help="Optional file to save context to")
    parser.add_argument("--format", choices=["plain", "markdown", "llm"], default="plain",
                      help="Output format: plain (default), markdown (for human reading), llm (minimal for LLM input)")
    
    args = parser.parse_args()
    
    context = query_vector_db(args.query, args.db, args.top)
    
    # Apply additional formatting based on output format
    if args.format == "markdown":
        context = f"# Context for Query: '{args.query}'\n\n" + context
    elif args.format == "llm":
        # Simplified format for LLM consumption - remove similarity scores and extra formatting
        context = re.sub(r'\[Similarity: \d+\.\d+\]\n\n', '', context)
        context = re.sub(r'={50}', '---', context)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(context)
        print(f"Context saved to {args.output}")
    else:
        print(context)

if __name__ == "__main__":
    main()