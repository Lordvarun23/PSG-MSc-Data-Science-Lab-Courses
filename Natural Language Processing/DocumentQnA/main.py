import streamlit as st
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import re
from io import BytesIO

ckpt = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(ckpt)
qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

def chunk_text(text):
    return text.split(".")

def get_document_embeddings(documents):
  embeddings = []
  model = SentenceTransformer("all-MiniLM-L6-v2")
  for i in range(len(documents)):
    emd = model.encode(documents[i]).tolist()
    embeddings.append(emd)
  return embeddings

def get_query_embeddings(query):
  model = SentenceTransformer("all-MiniLM-L6-v2")
  emb = model.encode(query).tolist()
  return emb

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    with BytesIO(uploaded_file.read()) as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text

# Function to answer user questions
def answer_question(question, pdf_text):
    re = qa({'question': question, "context": pdf_text})

    return re['answer']


def chromadb_results(name,embeddings,id,chunks,question):

    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.get_collection(name=re.sub(r'[^a-zA-Z0-9\s]', '', name)[:4])
    except:
        collection = chroma_client.get_or_create_collection(name=re.sub(r'[^a-zA-Z0-9\s]', '', name)[:4])
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=id
        )

    query_embedding = get_query_embeddings(question)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    return results

# Streamlit UI
def main():
    st.title("PDF Chatbot")

    # File uploader
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    #print(uploaded_file.name)
    if uploaded_file is not None:
        # Display PDF content
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Define chunk parameters
        chunk_size = 1000  # adjust as needed
        chunk_overlap = 200  # adjust as needed

        # Chunk the text
        chunks = chunk_text(pdf_text)
        embeddings = get_document_embeddings(chunks)
        id = [str(i) for i in range(len(embeddings))]

        st.subheader("Chatbot Interface")
        question = st.text_input("Ask a question:")

        results = chromadb_results(uploaded_file.name,embeddings,id,chunks,question)

        context = " ".join(results['documents'][0])
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', context)

        if st.button("Submit"):
            if question:
                st.text("Retreived text from the pdf: ")
                st.text(clean_text)
                response = answer_question(question, clean_text)
                st.text("Response:")
                st.write(response)

if __name__ == "__main__":
    main()