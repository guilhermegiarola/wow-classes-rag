import json
import requests
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
from markdown_text_clean import clean_text

try:
    from api.utils import web_scrape_data, generate_embedding_vector, split_into_chunks
except ImportError:
    from utils import web_scrape_data, generate_embedding_vector, split_into_chunks

load_dotenv()

api_key = os.getenv('FUELIX_API_KEY')
api_url = os.getenv('FUELIX_COMPLETIONS_URL', 'https://api.fuelix.ai/v1/chat/completions')

if not api_key:
    raise ValueError("FUELIX_API_KEY not found in environment variables. Please create a .env file.")

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

def get_article_content(article_content: str, debug_filename='debug_extracted_text.txt'):
    # Use path relative to this file's location
    query_txt_path = os.path.join(os.path.dirname(__file__), 'query.txt')
    
    with open(query_txt_path, 'r', encoding='utf-8') as f:
        query_template = f.read()
    
    llm_query = query_template.format(article_content=article_content)

    llm_response = generate_llm_response(llm_query)
    return llm_response

def generate_llm_response(llm_query: str):
    try:
        response = requests.post(api_url, headers=headers, json={'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': llm_query}]})
        response.raise_for_status()
        
        response_json = response.json()
        print(f"LLM Response status: {response.status_code}")
        
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
    db_path = os.path.join(os.path.dirname(__file__), 'milvus.db')
    milvus_client = MilvusClient(uri=db_path)
    collection_name = 'icy_veins'

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
        print(f"Collection {collection_name} dropped")
    
    milvus_client.create_collection(
        collection_name=collection_name, 
        dimension=1536,
        metric_type="IP",
        consistency_level="Bounded",
    )
    print(f"Collection {collection_name} created")

    print("Generating vectorized knowledge base from class files...")
    
    # Use paths relative to this file's location
    base_dir = os.path.dirname(__file__)
    classes_dir = os.path.join(base_dir, 'classes')
    classes_json_path = os.path.join(base_dir, 'classes.json')
    
    if not os.path.exists(classes_dir):
        return {
            'message': f'Error: classes directory not found at {classes_dir}. Run generate_knowledge_base() first.',
            'success': False
        }
    
    with open(classes_json_path, 'r') as f:
        classes = json.load(f)
    
    embeddings_data = []
    successful = 0
    failed = 0
    total_chunks = 0
    global_chunk_id = 0  # Auto-incrementing ID for all chunks
    
    for class_key in classes.keys():
        file_path = f"{classes_dir}/{class_key}.txt"
        
        if not os.path.exists(file_path):
            print(f"⚠ File not found: {file_path}, skipping...")
            failed += 1
            continue
        
        print(f"\nProcessing: {class_key}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content or len(content) < 100:
                print(f"✗ Content too short for {class_key}, skipping...")
                failed += 1
                continue
            
            chunks = split_into_chunks(content, chunk_size=1000, overlap=100)
            print(f"Processing {len(chunks)} chunks...")
            
            chunks_processed = 0
            response = generate_embedding_vector(chunks)
            if response and 'data' in response and response['data']:
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        embedding = response['data'][chunk_idx]['embedding']
                        
                        embeddings_data.append({
                            'id': global_chunk_id,  # Auto-incrementing integer ID
                            'class_name': class_key,
                            'chunk_index': chunk_idx,
                            'chunk_text': chunk,
                            'chunk_length': len(chunk),
                            'vector': embedding,  # MilvusClient uses 'vector' not 'embedding'
                            'file_path': file_path
                        })
                        
                        global_chunk_id += 1
                        chunks_processed += 1
                        
                    except Exception as e:
                        print(f"✗ Error processing chunk {chunk_idx}: {e}")
                        failed += 1
                        continue
            if chunks_processed > 0:
                print(f"✓ Processed {chunks_processed}/{len(chunks)} chunks")
                successful += 1
                total_chunks += chunks_processed
            else:
                print(f"✗ Failed to process chunks")
                failed += 1
                continue
            
        except Exception as e:
            print(f"✗ Unexpected error processing {class_key}: {e}")
            failed += 1
            continue
    
    # Save embeddings to Milvus database
    if embeddings_data:
        print(f"\n Inserting {len(embeddings_data)} chunks into Milvus...")
        milvus_client.insert(collection_name=collection_name, data=embeddings_data)
        print(f"✓ Successfully saved {len(embeddings_data)} chunks to database")
    else:
        print("No embeddings data to save!")
    
    return {
        'id': collection_name,
        'message': f'Vectorized knowledge base generated! Success: {successful}, Failed: {failed}, Total chunks: {total_chunks}',
        'successful_classes': successful,
        'failed_classes': failed,
        'total_classes': len(classes),
        'total_chunks': total_chunks,
        'chunks_in_db': len(embeddings_data)
    }

def generate_knowledge_base():
    """Scrape all classes and save each to a separate file"""
    # Use paths relative to this file's location
    base_dir = os.path.dirname(__file__)
    classes_json_path = os.path.join(base_dir, 'classes.json')
    classes_dir = os.path.join(base_dir, 'classes')
    
    # Load classes from JSON
    with open(classes_json_path, 'r') as f:
        classes = json.load(f)
    
    # Create classes directory if it doesn't exist
    os.makedirs(classes_dir, exist_ok=True)
    
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
            
            output_file = os.path.join(classes_dir, f"{class_key}.txt")
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
    db_path = os.path.join(os.path.dirname(__file__), 'milvus.db')
    milvus_client = MilvusClient(uri=db_path)
    
    # Generate embedding for the query
    embedding_response = generate_embedding_vector([query_text])
    
    # Extract the actual embedding vector from the response
    if not embedding_response or 'data' not in embedding_response or not embedding_response['data']:
        return {'message': 'Error generating query embedding', 'success': False}
    
    query_embedding = embedding_response['data'][0]['embedding']
    
    # Search for similar documents (fetch more results for better coverage)
    search_res = milvus_client.search(
        collection_name='icy_veins', 
        data=[query_embedding],
        limit=10,  # Fetch top candidates
        search_params={
            'metric_type': 'IP',
            'params': {}
        },
        output_fields=['class_name', 'chunk_text', 'chunk_index'],
    )

    print("Search results: ", search_res)
    
    # Check if we have results
    if not search_res[0]:
        return {'message': 'No results found', 'success': False}
    
    # Take top results regardless of class (up to 8 chunks)
    top_results = search_res[0][:8]
    
    # Group results by class to show which classes are relevant
    classes_involved = {}
    for result in top_results:
        class_name = result['class_name']
        if class_name not in classes_involved:
            classes_involved[class_name] = []
        classes_involved[class_name].append(result)
    
    # Build context organized by class
    context_parts = []
    for class_name, results in classes_involved.items():
        context_parts.append(f"\n=== {class_name} ===")
        for result in results:
            context_parts.append(f"{result['chunk_text']}")
    
    context = "\n".join(context_parts)
    context = context.replace('\n', ' ').replace('\r', ' ')
    context = clean_text(context)
    
    # Determine primary class (for metadata)
    primary_class = search_res[0][0]['class_name']
    primary_score = search_res[0][0]['distance']
    classes_list = list(classes_involved.keys())
    
    llm_query = f"""You are a World of Warcraft class guide expert. 
    Answer the user's question based on the provided context.

    Context from relevant class guides:
    {context}

    User Question:
    {query_text}

    Instructions:
    - Answer based ONLY on the provided context
    - If multiple classes are mentioned, address each one relevant to the question
    - Format your response in clear, well-organized markdown
    - If the context doesn't contain enough information, say so
    - Be concise but comprehensive
    """

    llm_response = generate_llm_response(llm_query)

    return {
        'message': 'Results retrieved', 
        'response': llm_response,
        'primary_class': primary_class,
        'classes_involved': classes_list,
        'confidence_score': primary_score
    }