from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set your API key here
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_llm():
    """Create and return a Google Gemini model instance"""
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    except Exception as e:
        print(f"Error creating LLM: {str(e)}")
        return None

def generate_response(query, context):
    """Generate a response using the LLM with provided context"""
    llm = create_llm()
    if not llm:
        return "LLM could not be initialized. Please check your API key and environment."
    
    try:
        # Format a prompt that includes both the context and query
        prompt = f"""
Based on the following context, answer the user's question.

CONTEXT:
{context}

USER QUESTION:
{query}

If the context doesn't contain relevant information to answer the question, 
please say "I don't have enough information to answer that question." 
Answer concisely and directly based only on the provided context.
"""
        
        # Call the LLM and return its response
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}" 