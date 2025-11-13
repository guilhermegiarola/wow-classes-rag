import json
import requests
import os
from dotenv import load_dotenv
from utils import web_scrape_data, generate_embedding_vector, split_into_chunks
from pymilvus import MilvusClient
from markdown_text_clean import clean_text

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
api_key = os.getenv('FUELIX_API_KEY')
api_url = os.getenv('FUELIX_API_URL', 'https://api.fuelix.ai/v1/chat/completions')

if not api_key:
    raise ValueError("FUELIX_API_KEY not found in environment variables. Please create a .env file.")

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

def get_article_content(article_content: str, debug_filename='debug_extracted_text.txt'):
    #get the article content from the soup
    
    with open('query.txt', 'r', encoding='utf-8') as f:
        query_template = f.read()
    
    # Insert article content into the template
    llm_query = query_template.format(article_content=article_content)

    llm_response = generate_llm_response(llm_query)
    return llm_response

def generate_llm_response(llm_query: str):
    #use the fuelixapi to generate a response to the llm query
    try:
        response = requests.post(api_url, headers=headers, json={'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': llm_query}]})
        response.raise_for_status()
        
        response_json = response.json()
        print(f"LLM Response status: {response.status_code}")
        
        # Extract the message content
        if 'choices' in response_json and len(response_json['choices']) > 0:
            message_content = response_json['choices'][0]['message']['content']
            return {'message': message_content}
        else:
            print(f"Unexpected API response format: {response_json}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing LLM response: {e}")
        return None

def generate_vectorized_knowledge_base():
    milvus_client = MilvusClient(uri="./milvus.db")
    collection_name = 'icy_veins'

    milvus_client.create_collection(
        collection_name=collection_name, 
        dimension=1536,
        metric_type="IP",
        consistency_level="Bounded",
    )
    
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
        print(f"Collection {collection_name} dropped")

    """Generate embeddings from already scraped class files"""
    print("Generating vectorized knowledge base from class files...")
    
    # Create classes directory if it doesn't exist
    classes_dir = 'classes'
    if not os.path.exists(classes_dir):
        return {
            'message': 'Error: classes directory not found. Run generate_knowledge_base() first.',
            'success': False
        }
    
    # Load class keys
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    
    embeddings_data = []
    successful = 0
    failed = 0
    total_chunks = 0
    
    # Process each class file
    for class_key in classes.keys():
        file_path = f"{classes_dir}/{class_key}.txt"
        
        if not os.path.exists(file_path):
            print(f"⚠ File not found: {file_path}, skipping...")
            failed += 1
            continue
        
        print(f"\nProcessing: {class_key}")
        
        try:
            # Read the class content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content or len(content) < 100:
                print(f"✗ Content too short for {class_key}, skipping...")
                failed += 1
                continue
            
            # Split content into chunks
            chunks = split_into_chunks(content, chunk_size=1000, overlap=100)
            print(f"Processing {len(chunks)} chunks...")
            
            # Generate embeddings for each chunk
            chunks_processed = 0
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    response = generate_embedding_vector(chunk)
                    
                    if response and 'data' in response and response['data']:
                        # Extract embedding and store
                        embedding = response['data'][0]['embedding']
                        
                        embeddings_data.append({
                            'class_name': class_key,
                            'chunk_id': f"{class_key}_chunk_{chunk_idx}",
                            'chunk_index': chunk_idx,
                            'chunk_text': chunk,
                            'chunk_length': len(chunk),
                            'embedding': embedding,
                            'file_path': file_path
                        })
                        
                        chunks_processed += 1
                        
                except Exception:
                    continue
            
            if chunks_processed > 0:
                print(f"✓ Processed {chunks_processed}/{len(chunks)} chunks")
                successful += 1
                total_chunks += chunks_processed
            else:
                print(f"✗ Failed to process chunks")
                failed += 1
            
        except Exception as e:
            print(f"✗ Unexpected error processing {class_key}: {e}")
            failed += 1
            continue
    
    # Save embeddings to file
    if embeddings_data:
        milvus_client.insert(collection_name=collection_name, data=embeddings_data)
        print(f"\n✓ Saved {total_chunks} chunks from {successful} classes")
    
    return {
        'message': f'Vectorized knowledge base generated! Success: {successful}, Failed: {failed}, Total chunks: {total_chunks}',
        'successful_classes': successful,
        'failed_classes': failed,
        'total_classes': len(classes),
        'total_chunks': total_chunks,
        'embeddings_file': 'embeddings.json'
    }

def retrieve_most_relevant_documents(query: str):
    #TODO: retrieve the most relevant documents from the vectorized knowledge base
    return {'message': 'Most relevant documents retrieved!'}

def generate_knowledge_base():
    """Scrape all classes and save each to a separate file"""
    # Load classes from JSON
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    
    # Create classes directory if it doesn't exist
    os.makedirs('classes', exist_ok=True)
    
    # URL endings to append
    url_endings = ['guide', 'rotation-cooldowns-abilities']
    
    results = {}
    
    # Process each class
    for class_key, base_url in classes.items():
        print(f"\n{'='*80}")
        print(f"Processing: {class_key}")
        print(f"{'='*80}")
        
        combined_content = []
        for idx, ending in enumerate(url_endings, 1):
            full_url = f"{base_url}{ending}"
            print(f"\n=== Scraping URL {idx}/{len(url_endings)}: {full_url} ===")
            
            try:
                content = web_scrape_data(full_url)
                combined_content.append(f"\n\n[SOURCE {idx}]\nURL: {full_url}\n\n{content}")
            except Exception as e:
                print(f"Error scraping {full_url}: {e}")
                continue
        
        if not combined_content:
            print(f"No content scraped for {class_key}, skipping...")
            continue
        
        article_content = '\n\n---\n\n'.join(combined_content)
        
        print(f"\nAnalyzing content for {class_key}...")
        try:
            llm_result = get_article_content(article_content, debug_filename=None)
            
            if not llm_result or 'message' not in llm_result or not llm_result['message']:
                print(f"✗ Invalid LLM response for {class_key}, skipping file write...")
                continue
            
            output_file = f"classes/{class_key}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(llm_result['message'])
            
            print(f"✓ Saved to {output_file}")
            results[class_key] = output_file
            
        except Exception as e:
            print(f"✗ Error processing LLM response for {class_key}: {e}")
            print(f"Existing file (if any) will be preserved.")
            continue
    
    return {
        'message': 'All classes processed successfully',
        'files': results
    }

def retrieve_answers(query_text: str):
    #TODO: retrieve the answers from the most relevant documents
    milvus_client = MilvusClient(uri="./milvus.db")
    
    # Generate embedding for the query
    embedding_response = generate_embedding_vector(query_text)
    
    # Extract the actual embedding vector from the response
    if not embedding_response or 'data' not in embedding_response or not embedding_response['data']:
        return {'message': 'Error generating query embedding', 'success': False}
    
    query_embedding = embedding_response['data'][0]['embedding']
    
    # Search for similar documents
    search_res = milvus_client.search(
        collection_name='icy_veins', 
        data=[query_embedding],  # Pass the raw embedding vector
        limit=4,
        search_params={
            'metric_type': 'IP',
            'params': {}
        },
        output_fields=['class_name', 'chunk_text'],
    )
    
    context = "\n".join([f"[{result['class_name']}] {result['chunk_text']}" for result in search_res[0]])
    # Clean out newline markers before applying clean_text
    context = context.replace('\n', ' ').replace('\r', ' ')
    context = clean_text(context)
    
    llm_query = f"""
        You are a comprehensive World of Warcraft class guide analyzer. 
        The context is:
        {context}
        The query is:
        {query_text}
        Please extract and organize all information in well-formatted markdown.
        Try to answer the query based on the context, with focus on the query.
    """

    llm_response = generate_llm_response(llm_query)

    return {'message': 'Results retrieved', 'response': llm_response}