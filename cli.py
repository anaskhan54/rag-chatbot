import os
import sys
from vectorize_pdf import vectorize_pdf
from agent import process_query
from llm import generate_response
import shutil
import textwrap

def print_colored(text, color):
    """Print colored text to the terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def print_boxed(title, content, color="blue"):
    """Print content in a box with a title."""
    width = shutil.get_terminal_size().columns - 4
    print_colored("\n┌" + "─" * (width - 2) + "┐", color)
    print_colored(f"│ {title.center(width - 4)} │", color)
    print_colored("├" + "─" * (width - 2) + "┤", color)
    
    # Wrap text to fit inside the box
    wrapper = textwrap.TextWrapper(width=width-4, break_long_words=False, replace_whitespace=False)
    wrapped_lines = []
    for line in content.split('\n'):
        if line.strip():
            wrapped_lines.extend(wrapper.wrap(line))
        else:
            wrapped_lines.append('')
    
    for line in wrapped_lines:
        padding = width - 4 - len(line)
        print_colored(f"│ {line}{' ' * padding} │", color)
    
    print_colored("└" + "─" * (width - 2) + "┘", color)

def main():
    print_colored("=== RAG-Powered Multi-Agent Q&A System ===", "cyan")
    
    # Step 1: Get PDF file path
    while True:
        pdf_path = input("Enter path to PDF file: ").strip()
        if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            break
        else:
            print_colored("File not found or not a PDF. Please try again.", "red")
    
    # Step 2: Copy the file to doc.pdf and vectorize it
    try:
        with open(pdf_path, 'rb') as src_file:
            with open('doc.pdf', 'wb') as dest_file:
                dest_file.write(src_file.read())
        
        print_colored("Processing document...", "yellow")
        vectorize_pdf()
        print_colored("Document ready.", "green")
    except Exception as e:
        print_colored(f"Error: {str(e)}", "red")
        sys.exit(1)
    
    # Step 3: Enter query loop
    print_colored("Type 'exit' to end or 'help' for examples.", "green")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
            
        if query.lower() == 'help':
            print_boxed("EXAMPLES", 
                       "1. 'What is this document about?'\n" +
                       "2. 'calculate 25 * 4 / 10'\n" +
                       "3. 'define algorithm'", 
                       "cyan")
            continue
        
        try:
            # Process the query
            result = process_query(query, generate_response)
            
            # Display tool used and answer directly
            tool_name = result["tool"].upper()
            print_colored(f"[{tool_name}]", "magenta")
            print_boxed("ANSWER", result["answer"], "green")
            
        except Exception as e:
            print_colored(f"Error: {str(e)}", "red")

if __name__ == "__main__":
    main()
