import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
# from langchain.vectorstores import Chroma
from huggingface_hub import login
from langchain_community.vectorstores import Chroma
import bs4
import chromadb
from transformers import pipeline
from torch import bfloat16
import os


INFO = None
with open(r"C:\Users\advay\Hackalytics\VQ-Trans\alltext.txt", "r", encoding="utf-8") as f:
    INFO = f.read()


def scrape_website(url):


    docs = INFO
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # chunk size (characters)
        chunk_overlap=200, # chunk overlap (characters)
        add_start_index=True, # track index in original document
    )
    all_splits = text_splitter.split_text(docs)
    return all_splits

def create_vectorstore(all_splits):
    
    # Create embeddings using a sentence transformer model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    vectorstore = Chroma(embedding_function=embeddings)
    vectorstore.add_documents(documents=all_splits)
    
    # Persist the database
    vectorstore.persist()
    return vectorstore
from langchain.chat_models import init_chat_model
import getpass
os.environ["OPENAI_API_KEY"] = ""
llm = init_chat_model("o1-mini", model_provider="openai")

_qa_chain = None
_transform_chain = None

prompt_template = """You are a physical therapy expert. Provide your best response to the patient describing an issue. 
Use the following text as context to respond. 

CONTEXT: ```{context}```

Q: \"{question}\"

RULES:
1. Provide direct instructions on the recommended mototion for the patient. 
2. Provide the motion for the single most appropriate exercise, not multiple exercises.
3. Followup with a short description of warnings or notes, like sets and reps.

A: \"Here's the recommended motion to address your problem: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

transform_template = """You are an exercise instruction parser. Extract only the physical movement instructions from the given text and convert them to third person format.

ORIGINAL TEXT: \"{text}\"

RULES:
1. Use "the person" or "they" instead of "you".
2. Only keep the movement itself and sequence, no other information.
3. Remove subjective modifiers like "comfortable range of motion" or "carefully".
4. Be concise and use simple and explicit language with layman terms, not medical terms like "dorsiflexion". 
5. Do not include movement options, like "sits or lies down." Simply describe one such option.
6. Ignore any supplementary information about benefits, warnings, notes, sets, reps, or duration of movement.
7. Keep your response length 20 words or less, and use commas (not periods) to separate phrases.
8. Use a response format similar to the example provided.

EXAMPLE: \"The person starts off in an upright position with both arms extended out by his sides, they then bring their arms down to their body and claps their hands together, after this they walk down and to the left where they proceed to sit on a seat.\"

Third person instructions: The person """

TRANSFORM_PROMPT = PromptTemplate(
    template=transform_template,
    input_variables=["text"]
)

def setup_rag_pipeline(vectorstore):
    global _qa_chain
    global _transform_chain
    """Set up the RAG pipeline with Mistral-7B."""
    
    # Create prompt template
    
    # Create and return the RAG chain
    if _qa_chain is None:
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Return top 3 most relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )

    
    if _transform_chain is None:
        _transform_chain = LLMChain(
            llm=llm,
            prompt=TRANSFORM_PROMPT
        )

    def combined_chain(query):
        # First get the QA response
        qa_result = _qa_chain(query)
        
        # Then transform to third person
        transformed = _transform_chain.run(qa_result['result'])
        
        return {
            'original_response': qa_result['result'],
            'third_person_instructions': transformed
        }
    
    
    return combined_chain

from langchain_core.documents import Document

# Example usage
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"  # Replace with your target URL

# Scrape website content
content = scrape_website(url)
documents = [Document(page_content=chunk) for chunk in content]
# Create vectorstore
vectorstore = create_vectorstore(documents)
def main():
    while True:
    # Setup RAG pipeline
        rag_chain = setup_rag_pipeline(vectorstore)
        
        # Example query
        print("Whats up?")
        question = input()
        response = rag_chain(question)
        # print(f"Question: {question}")
        print("FIRST OUTPUT")
        print(response['original_response'])
        print("FINAL OUTPUT")
        print(response['third_person_instructions'])


if __name__ == "__main__":
    main()