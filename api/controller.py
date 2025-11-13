from fastapi import FastAPI
import service 

app = FastAPI()

@app.get('/retrieve-answers')
def retrieve_answers(query_text: str):
    return service.retrieve_answers(query_text)

@app.get('/generate-knowledge-base')
def web_scrape_data():
    return service.generate_knowledge_base()

@app.get('/generate-embedding-vector')
def generate_embedding_vector():
    return service.generate_vectorized_knowledge_base()

def read_root():
    return {'message': 'Hello World!'}
