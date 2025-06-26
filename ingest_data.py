import os
import fitz  
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# --- Parsing Functions ---

def parse_pdf(file_path: str) -> List[Dict]:
    """
    Parses a PDF document, extracting text from each page.
    Returns a list of dictionaries, where each dictionary represents a page.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return []

    doc = fitz.open(file_path)
    pages_content = []
    for page_num, page in enumerate(doc):
        pages_content.append({
            "source": f"{os.path.basename(file_path)} - Page {page_num + 1}",
            "content": page.get_text()
        })
    doc.close()
    return pages_content

def parse_markdown(file_path: str) -> List[Dict]:
    """
    Parses a Markdown file.
    Returns a list containing a single dictionary for the entire file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [{
        "source": os.path.basename(file_path),
        "content": content
    }]

def parse_text(file_path: str) -> List[Dict]:
    """
    Parses a plain text file.
    Returns a list containing a single dictionary for the entire file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [{
        "source": os.path.basename(file_path),
        "content": content
    }]

def scrape_web_page(url: str) -> List[Dict]:
    """
    Scrapes the textual content from a given URL.
    Returns a list containing a single dictionary for the web page.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove non-content tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Get clean text
        text = soup.get_text(separator='\n', strip=True)
        return [{"source": url, "content": text}]
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []

# --- Main Ingestion Logic ---

def load_documents(source_dir: str) -> List[Dict]:
    """
    Loads all supported documents from a directory.
    """
    all_docs = []
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if filename.endswith(".pdf"):
            all_docs.extend(parse_pdf(file_path))
        elif filename.endswith(".md"):
            all_docs.extend(parse_markdown(file_path))
        elif filename.endswith(".txt"):
            all_docs.extend(parse_text(file_path))
    return all_docs

