import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber
import shutil

# Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is missing. Check your .env file.")

genai.configure(api_key=API_KEY)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    
    if not text.strip():
        raise ValueError("Error: No text extracted from PDFs. The file may be scanned or protected.")
    
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("Error: No text chunks generated. Check PDF text extraction.")

    return chunks


def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("Error: No text chunks found. Cannot create vector store.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Ensure embeddings are generated correctly
    try:
        sample_embedding = embeddings.embed_query("test")
        if not sample_embedding or len(sample_embedding) == 0:
            raise ValueError("Error: Failed to generate embeddings. Check API key or model.")
    except Exception as e:
        raise ValueError(f"Error: Embedding generation failed - {e}")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Ensure embeddings are working
    try:
        sample_embedding = embeddings.embed_query("test")
        if not sample_embedding or len(sample_embedding) == 0:
            raise ValueError("Error: Failed to generate embeddings. Check API key or model.")
    except Exception as e:
        st.error(f"Error: Embedding generation failed - {e}")
        return

    # Try to load FAISS index, handle errors
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}. Rebuilding FAISS index...")
        # Rebuild FAISS if loading fails
        shutil.rmtree("faiss_index", ignore_errors=True)
        st.warning("FAISS index deleted. Please re-upload and process PDFs.")
        return

    if not docs:
        st.warning("No relevant documents found in the FAISS index.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
