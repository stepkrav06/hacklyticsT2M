from flask import Flask, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from huggingface_hub import login
import bs4
import chromadb
from transformers import pipeline
from torch import bfloat16
import os
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

# NEW: Initialize Flask app
app = Flask(__name__)

# Keep existing global variables and constants
INFO = None
with open(r"C:\Users\advay\Hackalytics\VQ-Trans\alltext.txt", "r", encoding="utf-8") as f:
    INFO = f.read()

_qa_chain = None
_transform_chain = None
os.environ["OPENAI_API_KEY"] = ""

# Keep existing prompt templates
prompt_template = """You are a physical therapy expert. Provide your best response to the patient describing an issue. 
Use the following text as context to respond. 

CONTEXT: ```{context}```

Q: \"{question}\"

RULES:
1. Provide direct, descriptive instructions on the recommended motion for the patient. 
2. Provide the motion for the single most appropriate exercise, not multiple exercises.
3. Followup with a short description of warnings or notes, like sets and reps.

A: \"Here's a detailed description for the recommended motion to address your problem: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

transform_template = """You are an exercise instruction parser. Extract the physical movement instructions from the given text and convert them to third person format.
INPUT: \"{text}\"

RULES:
1. Use "a person" or "they" instead of "you".
2a. Describe one single clear sequence of movements. 
2b. Parse only the first complete movement sequence if there are multiple present.
2c. Describe one single iteration of repeated movements.
2c. Limit to 3 sequential actions.
3. Omit actions involving sitting and laying on back unless essential.
4. Remove numbers, references to objects, and subjective modifiers like "relaxed shoulders" or "comfortable motion".
5. Do not provide options like "sit or stand." Just describe one option.
6. Use precise but non-technical, layman language as if talking to a robot.
7. Ignore additional information (benefits, notes, sets, reps, duration).
8. Keep the response length to 20 words, use commas to separate phrases, and do not add quotes or other formatting to the output.
9. Use sequential markers like "then" only when necessary for clarity.

EXAMPLE 1: \"A person starts off in an upright position with both arms extended out by his sides, they then bring their arms down to their body, and clap their hands together.\"
EXAMPLE 2: \"A person jogs in place, they then back up, and squat down.\"
EXAMPLE 3: \"A person rises from the ground, walks in a circle, and sits back down on the ground.\"
EXAMPLE 4: \"A person slightly crouches down and walks forward, then back, then around slowly.\"

OUTPUT FORMAT: A person [action 1], [action 2], [action 3]
"""

TRANSFORM_PROMPT = PromptTemplate(
    template=transform_template,
    input_variables=["text"]
)

# Keep existing functions
def scrape_website(url):
    docs = INFO
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_text(docs)
    return all_splits

def create_vectorstore(all_splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Chroma(embedding_function=embeddings)
    vectorstore.add_documents(documents=all_splits)
    vectorstore.persist()
    return vectorstore

def setup_rag_pipeline(vectorstore):
    global _qa_chain
    global _transform_chain
    
    if _qa_chain is None:
        llm = init_chat_model("gpt-4o", model_provider="openai")
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    if _transform_chain is None:
        llm = init_chat_model("o1-mini", model_provider="openai")
        _transform_chain = LLMChain(
            llm=llm,
            prompt=TRANSFORM_PROMPT
        )

    def combined_chain(query):
        qa_result = _qa_chain(query)
        transformed = _transform_chain.run(qa_result['result'])
        return {
            'original_response': qa_result['result'],
            'third_person_instructions': transformed
        }
    
    return combined_chain

# NEW: Initialize vectorstore at startup
content = scrape_website("dummy_url")  # URL not used in current implementation
documents = [Document(page_content=chunk) for chunk in content]
vectorstore = create_vectorstore(documents)
rag_chain = setup_rag_pipeline(vectorstore)

# NEW: Flask endpoint
@app.route('/get_exercise', methods=['POST'])
def get_exercise():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question']
        response = rag_chain(question)
        
        # return jsonify({
        #     'original_response': response['original_response'],
        #     'third_person_instructions': response['third_person_instructions']
        # })

        response_api = requests.post('http://localhost:5000/generate_motion', json={'text': response['third_person_instructions']})
        
        return jsonify({
            'message': 'Motion generated successfully',
            'animation_id': response_api.json()['animation_id'],
            'question': response['original_response'],
            'prompt': response['third_person_instructions']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_animation/<int:animation_id>', methods=['GET'])
def get_animation(animation_id):
    """Endpoint to retrieve the generated animation"""
    try:
        gif_path = os.path.join('generated', f'animation_{animation_id}.gif')
        if not os.path.exists(gif_path):
            return jsonify({'error': 'Animation not found'}), 404
            
        return send_file(
            gif_path,
            mimetype='image/gif',
            as_attachment=False,
            download_name=f'animation_{animation_id}.gif'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == "__main__":
    # Initialize everything at startup
    app.run(host='0.0.0.0', port=3000)