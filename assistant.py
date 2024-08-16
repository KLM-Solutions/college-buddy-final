import streamlit as st
from docx import Document
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random
import sqlite3
import pandas as pd
from difflib import SequenceMatcher
import os
from langsmith import Client, trace
import functools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import asyncio
from typing import AsyncIterable

# Load environment variables
load_dotenv()

# Access your API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
INDEX_NAME = "college-buddy"

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "College-Buddy-Assistant"

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
langsmith_client = Client(api_key=LANGCHAIN_API_KEY)
chat = ChatOpenAI(model_name="gpt-4", temperature=0.3)
embeddings = OpenAIEmbeddings()

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# List of example questions
EXAMPLE_QUESTIONS = [
    "What are the steps to declare a major at Texas Tech University",
    "What are the GPA and course requirements for declaring a major in the Rawls College of Business?",
    "How can new students register for the Red Raider Orientation (RRO)",
    "What are the key components of the Texas Tech University Code of Student Conduct",
    "What resources are available for students reporting incidents of misconduct at Texas Tech University",
    "What are the guidelines for amnesty provisions under the Texas Tech University Code of Student Conduct",
    "How does Texas Tech University handle academic misconduct, including plagiarism and cheating",
    "What are the procedures for resolving student misconduct through voluntary resolution or formal hearings",
    "What are the rights and responsibilities of students during the investigative process for misconduct at Texas Tech University",
    "How can students maintain a healthy lifestyle, including nutrition and fitness, while attending Texas Tech University"
]

# Database functions
@st.cache_resource
def get_database_connection():
    conn = sqlite3.connect('college_buddy.db', check_same_thread=False)
    return conn

def init_db(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, title TEXT, tags TEXT, links TEXT)''')
    conn.commit()

def load_initial_data():
    conn = get_database_connection()
    data = [
        (1, "TEXAS TECH", "Universities, Texas Tech University, College Life, Student Wellness, Financial Tips for Students, Campus Activities, Study Strategies", "https://www.ttu.edu/"),
        # ... (rest of the initial data)
    ]
    c = conn.cursor()
    c.executemany("INSERT OR REPLACE INTO documents (id, title, tags, links) VALUES (?, ?, ?, ?)", data)
    conn.commit()

def insert_document(title, tags, links):
    if tags.strip() and links.strip():
        conn = get_database_connection()
        c = conn.cursor()
        c.execute("INSERT INTO documents (title, tags, links) VALUES (?, ?, ?)",
                  (title, tags, links))
        conn.commit()
        return True
    return False

def get_all_documents():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, tags, links FROM documents WHERE tags != '' AND links != ''")
    return c.fetchall()

# Utility functions
def safe_run_tree(name, run_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with trace(name=name, run_type=run_type, client=langsmith_client) as run:
                    result = func(*args, **kwargs)
                    run.end(outputs={"result": str(result)})
                    return result
            except Exception as e:
                st.error(f"Error in LangSmith tracing: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@safe_run_tree(name="get_embedding", run_type="llm")
def get_embedding(text):
    with get_openai_callback() as cb:
        embedding = embeddings.embed_query(text)
    return embedding

def upsert_to_pinecone(text, file_name, file_id):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "file_name": file_name,
            "file_id": file_id,
            "chunk_id": i,
            "chunk_text": chunk
        }
        index.upsert(vectors=[(f"{file_id}_{i}", embedding, metadata)])
        time.sleep(1)  # To avoid rate limiting

def query_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'chunk_text' in match['metadata']:
            contexts.append(match['metadata']['chunk_text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('file_name', 'unknown file')}")
    return " ".join(contexts)

@safe_run_tree(name="identify_intents", run_type="llm")
def identify_intents(query):
    system_message = SystemMessage(content="You are an intent identification assistant. Identify and provide only the primary intent or question within the given query.")
    human_message = HumanMessage(content=f"Identify the main intent or question within this query: {query}")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    
    intent = response.content.strip()
    return [intent] if intent else []

@safe_run_tree(name="generate_keywords_per_intent", run_type="llm")
def generate_keywords_per_intent(intents):
    intent_keywords = {}
    for intent in intents:
        system_message = SystemMessage(content="You are a keyword extraction assistant. Generate relevant keywords or phrases for the given intent.")
        human_message = HumanMessage(content=f"Generate 5-10 relevant keywords or phrases for this intent, separated by commas: {intent}")
        
        with get_openai_callback() as cb:
            response = chat([system_message, human_message])
        
        keywords = response.content.strip().split(',')
        intent_keywords[intent] = [keyword.strip() for keyword in keywords]
    return intent_keywords

def query_db_for_keywords(keywords):
    conn = get_database_connection()
    c = conn.cursor()
    query = """
    SELECT DISTINCT id, title, tags, links 
    FROM documents 
    WHERE tags LIKE ?
    """
    results = []
    for keyword in keywords:
        c.execute(query, (f'%{keyword}%',))
        for row in c.fetchall():
            score = sum(SequenceMatcher(None, keyword.lower(), tag.lower()).ratio() for tag in row[2].split(','))
            results.append((score, row))
    
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:3]

def query_for_multiple_intents(intent_keywords):
    intent_data = {}
    all_db_results = set()
    for intent, keywords in intent_keywords.items():
        db_results = query_db_for_keywords(keywords)
        new_db_results = [result for result in db_results if result[1][0] not in [r[1][0] for r in all_db_results]]
        all_db_results.update(new_db_results)
        pinecone_context = query_pinecone(" ".join(keywords))
        intent_data[intent] = {
            'db_results': new_db_results,
            'pinecone_context': pinecone_context
        }
    return intent_data

@safe_run_tree(name="generate_multi_intent_answer", run_type="llm")
def generate_multi_intent_answer(query, intent_data):
    context = "\n".join([f"Intent: {intent}\nDB Results: {data['db_results']}\nPinecone Context: {data['pinecone_context']}" for intent, data in intent_data.items()])
    max_context_tokens = 4000
    truncated_context = truncate_text(context, max_context_tokens)
    
    system_message = SystemMessage(content="""You are College Buddy, an AI assistant designed to help students with their academic queries. Your primary function is to analyze and provide insights based on the context of uploaded documents. Please adhere to the following guidelines:
1. Focus on addressing the primary intent of the query.
2. Provide accurate, relevant information derived from the provided context.
3. If the context doesn't contain sufficient information to answer the query, state this clearly.
4. Maintain a friendly, supportive tone appropriate for assisting students.
5. Provide concise yet comprehensive answers, breaking down complex concepts when necessary.
6. If asked about topics beyond the scope of the provided context, politely state that you don't have that information.
7. Encourage critical thinking by guiding students towards understanding rather than simply providing direct answers.
8. Respect academic integrity by not writing essays or completing assignments on behalf of students.
9. Suggest additional resources only if directly relevant to the primary query.
""")
    human_message = HumanMessage(content=f"Query: {query}\n\nContext: {truncated_context}")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    
    return response.content.strip()

@safe_run_tree(name="extract_keywords_from_response", run_type="llm")
def extract_keywords_from_response(response):
    system_message = SystemMessage(content="You are a keyword extraction assistant. Extract key terms or phrases from the given text.")
    human_message = HumanMessage(content=f"Extract 5-10 key terms or phrases from this text, separated by commas: {response}")
    
    with get_openai_callback() as cb:
        keyword_response = chat([system_message, human_message])
    
    keywords = keyword_response.content.strip().split(',')
    return [keyword.strip() for keyword in keywords]

@safe_run_tree(name="get_answer", run_type="chain")
def get_answer(query):
    intents = identify_intents(query)
    intent_keywords = generate_keywords_per_intent(intents)
    intent_data = query_for_multiple_intents(intent_keywords)
    initial_answer = generate_multi_intent_answer(query, intent_data)
    
    response_keywords = extract_keywords_from_response(initial_answer)
    all_keywords = list(set(intent_keywords[intents[0]] + response_keywords))
    
    expanded_intent_data = query_for_multiple_intents({query: all_keywords})
    final_answer = generate_multi_intent_answer(query, expanded_intent_data)
    
    return final_answer, expanded_intent_data, all_keywords

async def generate_streaming_response(answer: str) -> AsyncIterable[str]:
    words = answer.split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.05)  # Adjust the delay as needed

# Streamlit Interface
def main():
    st.set_page_config(page_title="College Buddy Assistant", layout="wide")

    if not LANGCHAIN_API_KEY:
        st.warning("LangSmith API key is not set. Some features may not work properly.")

    st.title("College Buddy Assistant")
    st.markdown("Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance. Feel free to ask me a question below.")

    # Initialize database connection
    conn = get_database_connection()
    init_db(conn)
    load_initial_data()

    # Sidebar for file upload and metadata
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload the Word Documents (DOCX)", type="docx", accept_multiple_files=True)
        if uploaded_files:
            total_token_count = 0
            for uploaded_file in uploaded_files:
                file_id = str(uuid.uuid4())
                text = extract_text_from_docx(uploaded_file)
                token_count = num_tokens_from_string(text)
                total_token_count += token_count
                upsert_to_pinecone(text, uploaded_file.name, file_id)
                st.text(f"Uploaded: {uploaded_file.name}")
                st.text(f"File ID: {file_id}")
            st.subheader("Uploaded Documents")
            st.text(f"Total token count: {total_token_count}")
        if st.button("View Database"):
            st.switch_page("pages/database.py")
        if st.button("Manage Database"):
            st.switch_page("pages/database.py")
   
    # Main content area
    st.header("Popular Questions")
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)

    for question in st.session_state.selected_questions:
        if st.button(question, key=question):
            st.session_state.current_question = question

    st.header("Ask Your Own Question")
    user_query = st.text_input("What would you like to know about the uploaded documents?")

   

    if st.button("Get Answer"):
        if user_query:
            st.session_state.current_question = user_query
        elif 'current_question' not in st.session_state:
            st.warning("Please enter a question or select a popular question before searching.")

    if 'current_question' in st.session_state:
        with st.spinner("Searching for the best answer..."):
            with trace(name="process_query", run_type="chain", client=langsmith_client) as run:
                answer, intent_data, keywords = get_answer(st.session_state.current_question)
                run.end(outputs={"answer": answer})
            
            st.subheader("Question:")
            st.write(st.session_state.current_question)
            st.subheader("Answer:")
            
            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            
            # Use asyncio to run the streaming response
            async def display_streaming_response():
                full_response = ""
                async for word in generate_streaming_response(answer):
                    full_response += word
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            # Run the streaming response
            asyncio.run(display_streaming_response())
            
            st.subheader("Related Keywords:")
            st.write(", ".join(keywords))
            
            st.subheader("Related Documents:")
            displayed_docs = set()
            for intent, data in intent_data.items():
                for score, doc in data['db_results']:
                    if doc[0] not in displayed_docs:
                        displayed_docs.add(doc[0])
                        with st.expander(f"Document: {doc[1]}"):
                            st.write(f"ID: {doc[0]}")
                            st.write(f"Title: {doc[1]}")
                            st.write(f"Tags: {doc[2]}")
                            st.write(f"Link: {doc[3]}")
                            
                            highlighted_tags = doc[2]
                            for keyword in keywords:
                                highlighted_tags = highlighted_tags.replace(keyword, f"**{keyword}**")
                            st.markdown(f"Matched Tags: {highlighted_tags}")
  
        # Add to chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append((st.session_state.current_question, answer))
        
        # Clear the current question
        del st.session_state.current_question

    # Add a section for displaying recent questions and answers
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.header("Recent Questions and Answers")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {q}"):
                st.write(f"A: {a}")

if __name__ == "__main__":
    with trace(name="College_Buddy_Assistant", client=langsmith_client) as root_run:
        main()
