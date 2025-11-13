from bs4 import BeautifulSoup
import requests

api_key = 'REDACTED_API_KEY'
api_url = 'https://api.fuelix.ai/v1/embeddings'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}
data = {
    'model': 'text-embedding-3-small',
    'dimensions': 1536,
    'encoding_format': 'float'
}

def extract_structured_content(content_element):
    """Extract content while preserving tables, lists, and hierarchy"""
    result = []
    
    # Process all children elements
    for element in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table', 'div'], recursive=True):
        # Skip if this element is inside another element we'll process
        if element.find_parent(['table', 'ul', 'ol']) and element.name not in ['table', 'ul', 'ol']:
            continue
            
        # Handle headings - use [HEADING] markers to avoid markdown conflicts
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = element.name[1]
            text = element.get_text(strip=True)
            if text:
                result.append(f"\n[H{level}] {text}\n")
        
        # Handle paragraphs
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short paragraphs
                result.append(text)
        
        # Handle unordered lists
        elif element.name == 'ul':
            items = element.find_all('li', recursive=False)
            for item in items:
                text = item.get_text(strip=True)
                if text:
                    result.append(f"- {text}")
        
        # Handle ordered lists
        elif element.name == 'ol':
            items = element.find_all('li', recursive=False)
            for idx, item in enumerate(items, 1):
                text = item.get_text(strip=True)
                if text:
                    result.append(f"{idx}. {text}")
        
        # Handle tables (preserve structure as text)
        elif element.name == 'table':
            result.append("\n[TABLE]")
            # Get headers
            headers = element.find_all('th')
            if headers:
                header_text = " | ".join([h.get_text(strip=True) for h in headers])
                result.append(header_text)
                result.append("-" * len(header_text))
            
            # Get rows
            rows = element.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells and not all(c.name == 'th' for c in cells):  # Skip header row
                    row_text = " | ".join([c.get_text(strip=True) for c in cells])
                    if row_text.strip():
                        result.append(row_text)
            result.append("[/TABLE]\n")
    
    return '\n'.join(result)

def web_scrape_data(url, save_debug=False):
    """Scrape web content using BeautifulSoup
    
    Args:
        url: URL to scrape
        save_debug: If True, save raw HTML to debug_scraped.html (default: False)
    """
    print(f"Fetching URL: {url}")
    
    # Add headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Fetch the page
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Debug: Save the raw HTML (optional)
    if save_debug:
        with open('debug_scraped.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved raw HTML to debug_scraped.html for inspection")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
        element.decompose()
    
    # Try multiple selectors for content
    article_content = (
        soup.find('div', class_='guide-content') or 
        soup.find('div', class_='text') or
        soup.find('div', id='guide-body') or
        soup.find('div', id='main-contents') or
        soup.find('article') or 
        soup.find('main') or 
        soup.find('div', class_='content') or 
        soup.find('body')
    )
    
    if article_content:
        # Extract structured content with better preservation
        structured_content = extract_structured_content(article_content)
        
        # Check if we got meaningful content
        if len(structured_content) > 100:
            print(f"Extracted {len(structured_content)} characters of content")
            return structured_content
    
    # Fallback: get all text
    fallback_content = soup.get_text(separator='\n\n', strip=True)
    print(f"Using fallback extraction: {len(fallback_content)} characters")
    return fallback_content

def split_into_chunks(text: str, chunk_size: int = 2000, overlap: int = 200):
    """Split text into overlapping chunks - optimized version
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk (if we're at the end, we're done)
        if end >= text_length:
            break
        
        # Move forward with overlap
        start = end - overlap
    
    return chunks

def generate_embedding_vector(chunk: str):
    """Generate embedding for a single chunk"""
    data['input'] = chunk
    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()