import re
from get_context import query_vector_db
import requests
import math

def use_calculator(query):
    """Simple calculator tool that evaluates mathematical expressions"""
    # Extract the mathematical expression from the query
    expression = query.replace("calculate", "").replace("compute", "").strip()
    try:
        # Use safer eval with math module
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculator result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

def use_dictionary(query):
    """Dictionary tool that looks up word definitions"""
    # Extract the word to define
    match = re.search(r"define\s+(\w+)", query, re.IGNORECASE)
    if not match:
        word = query.replace("define", "").replace("what is", "").replace("?", "").strip()
    else:
        word = match.group(1)
    
    # Use a free dictionary API
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                meanings = data[0].get("meanings", [])
                if meanings:
                    definitions = []
                    for meaning in meanings[:2]:  # Limit to 2 meanings
                        part_of_speech = meaning.get("partOfSpeech", "")
                        definition = meaning.get("definitions", [{}])[0].get("definition", "")
                        if definition:
                            definitions.append(f"({part_of_speech}) {definition}")
                    
                    return f"Definition of '{word}':\n" + "\n".join(definitions)
        
        return f"Could not find definition for '{word}'"
    except Exception as e:
        return f"Error looking up definition: {str(e)}"

def process_query(query, llm_function):
    """
    Process a query using the appropriate tool or RAG pipeline.
    
    Args:
        query: User's question
        llm_function: Function to call LLM with context
        
    Returns:
        dict with:
        - tool: The tool/path used ("calculator", "dictionary", or "rag")
        - context: Any retrieved context (for RAG)
        - answer: The final answer
    """
    # Log the beginning of query processing
    print(f"Processing query: '{query}'")
    
    # Convert query to lowercase for keyword matching
    query_lower = query.lower()
    
    # Check for calculator keywords
    if any(keyword in query_lower for keyword in ["calculate", "compute", "sum of", "product of"]):
        print("Decision: Using calculator tool")
        answer = use_calculator(query)
        return {
            "tool": "calculator",
            "context": None,
            "answer": answer
        }
    
    # Check for dictionary keywords
    elif any(keyword in query_lower for keyword in ["define", "definition of", "meaning of", "what is the meaning of"]):
        print("Decision: Using dictionary tool")
        answer = use_dictionary(query)
        return {
            "tool": "dictionary",
            "context": None,
            "answer": answer
        }
    
    # Default to RAG pipeline
    else:
        print("Decision: Using RAG pipeline")
        # Get context from vector DB
        context = query_vector_db(query)
        
        # If we had an LLM, we would pass the context to it here
        if llm_function:
            answer = llm_function(query, context)
        else:
            # Fallback if no LLM function provided
            answer = f"Here is the relevant information I found:\n{context}"
            
        return {
            "tool": "rag",
            "context": context,
            "answer": answer
        } 