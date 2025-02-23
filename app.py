import os
import gradio as gr
from PyPDF2 import PdfReader
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import time  # For debugging and timeout handling

# Load API Key from Hugging Face Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Read from environment variables

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in Hugging Face Secrets.")

# Load the English-to-Urdu translation model
def translate_to_urdu(text):
    model_name = "Helsinki-NLP/opus-mt-en-ur"  # Pre-trained English-to-Urdu model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize and translate
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text

@lru_cache(maxsize=None)
def initialize_system():
    # Read PDFs
    text = ""
    for file in os.listdir():
        if file.endswith(".pdf"):
            with open(file, "rb") as f:
                reader = PdfReader(f)
                text += "".join([page.extract_text() or "" for page in reader.pages])

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text)

    # Multilingual embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Load LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.1
    )

    return vector_store, llm

# Initialize the system with a timeout handler
try:
    vector_store, llm = initialize_system()
except Exception as e:
    print(f"Initialization failed: {e}")
    raise

# Prompt with context and query
def get_prompt(language):
    if language == "ur":
        return """
        آپ ایک قانونی ماہر ہیں جو پاکستانی قانونی دستاویزات کی بنیاد پر جوابات دیتے ہیں۔
        فراہم کردہ مواد کی بنیاد پر جواب دیں۔
        
        سیاق و سباق: {context}
        سوال: {query}
        براہ کرم اردو میں واضح اور تفصیلی جواب دیں۔ اگر معلومات نہ ہوں تو کہیں: "دستاویز میں موجود نہیں۔"
        جواب میں درج ذیل نکات شامل کریں:
        - قانونی دفعات
        - ممکنہ سزائیں
        - متعلقہ استثنیات یا شرائط
        """
    else:
        return """
        You are a legal expert answering based on Pakistan's legal documents.
        Answer strictly using the provided context.
        
        Context: {context}
        Question: {query}
        Please provide a detailed and structured answer. If unsure, say: "Not found in documents."
        Your answer should include:
        - Relevant legal provisions
        - Possible punishments
        - Any exceptions or conditions
        """

# Generate response with context and query
@lru_cache(maxsize=128)
def get_response(query):
    try:
        language = detect(query)
        prompt_template = get_prompt(language)

        # Step 1: Retrieve relevant documents (in English)
        relevant_docs = vector_store.similarity_search(query, k=5)  # Retrieve top 5 documents
        context = " ".join([doc.page_content for doc in relevant_docs])  # Combine into a single context

        # Step 2: Translate context to Urdu if the query is in Urdu
        if language == "ur":
            context = translate_to_urdu(context)  # Translate English context to Urdu

        # Step 3: Format the prompt with the retrieved context and query
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )
        formatted_prompt = PROMPT.format(context=context, query=query)

        # Step 4: Generate the response using the LLM
        response = llm.invoke(
            formatted_prompt,
            temperature=0.5,  # Increase temperature for more creative/elaborate responses
            max_tokens=1000  # Increase max_tokens for longer responses
        )

        return response.content  # Return the generated response

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Chat Interface
def chat_interface(message, history):
    return get_response(message)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🏛️ Pakistan Legal Assistant 🇵🇰")
    gr.ChatInterface(fn=chat_interface, examples=[
        "What is the punishment for theft?",
        "چوری کی سزا کیا ہے؟",
        "Explain Section 302 of PPC",
        "PECA 2016 کے تحت سائبر کرائم کی سزائیں؟"
    ])

if __name__ == "__main__":
    demo.launch()
