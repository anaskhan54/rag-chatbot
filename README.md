# RAG-Powered Multi-Agent Q&A System

A simple knowledge assistant that retrieves information from documents, generates answers, and uses a basic agentic workflow to route queries to appropriate tools.

## Architecture

This system consists of several components:

1. **Data Ingestion & Vectorization** - `vectorize_pdf.py`
   - Extracts text from PDF documents
   - Chunks text into manageable segments
   - Converts text chunks into vectors using Sentence Transformers
   - Stores vectors for later retrieval

2. **Vector Retrieval** - `get_context.py`
   - Loads the vector database
   - Converts queries to vectors using the same model
   - Finds relevant document chunks using cosine similarity
   - Returns formatted context

3. **Agent System** - `agent.py`
   - Routes queries to appropriate tools based on keywords
   - Handles calculator functionality for math expressions
   - Provides dictionary definitions through a free API
   - Falls back to RAG pipeline for other queries

4. **LLM Integration** - `llm.py`
   - Uses Google Generative AI (Gemini) through Langchain
   - Generates responses based on query and context

5. **CLI Interface** - `cli.py`
   - Interactive command-line interface for direct user interaction
   - Handles PDF selection, processing, and query loop
   - Displays tool selection and answers

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Google API key:
   ```python
   # In llm.py
   os.environ["GOOGLE_API_KEY"] = "your-api-key"
   ```

## Usage

The system is operated through a simple command-line interface:

```
python cli.py
```

### Workflow

1. When prompted, provide the path to your PDF document
2. The system will process and vectorize the document
3. You can then enter queries about the document content
4. Type 'help' to see example queries
5. Type 'exit' to end the session

### Query Types

The system supports different types of queries:

1. **Document Queries**: Ask questions about the document content
   - Example: "What is this document about?"
   - Uses the RAG (Retrieval-Augmented Generation) pipeline

2. **Calculations**: Perform mathematical operations
   - Example: "calculate 25 * 4 / 10"
   - Uses a built-in calculator tool

3. **Definitions**: Look up word definitions
   - Example: "define algorithm"
   - Uses a dictionary API

## Design Choices

- **Simple Branching Logic**: The agent uses keyword matching to determine which tool to use
- **Tool Selection**: Implemented calculator and dictionary tools as specified in requirements
- **LLM Integration**: Used Google's Gemini model through Langchain for easy integration
- **Minimal UI**: Clean CLI interface showing only essential information
- **Query Routing**: Each query is analyzed and routed to the appropriate tool automatically 