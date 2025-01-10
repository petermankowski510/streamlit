import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI  # Use this for chat models
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import timeit
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = (
    ""
)

# Sidebar contents
with st.sidebar:
    st.title("Scientium.ai API Framework 2025")
    st.markdown(
        """

    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    """
    )
    add_vertical_space(5)
    st.write("Copyright 2025: [Scientium.ai](https://www.youtube.com/watch?v=5FqEXniUmO0)")

def main():
    st.header("Custom ChatBot .PDF Uploader")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Safely display the name of the uploaded PDF
        st.write(f"Uploaded file: **{pdf.name}**")

        # Read the PDF content
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Define your embeddings
        embeddings = OpenAIEmbeddings()

        # If the FAISS index folder already exists, load it
        # Otherwise, build and save it
        if os.path.exists("faiss_index"):
            VectorStore = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )
            st.write("Embeddings loaded from disk.")
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local("faiss_index")
            st.write("Embeddings computation completed.")
    else:
        st.warning("Please upload a PDF file.")

    # Accept user questions/query
    query = st.text_input("Ask questions/analytics about your PDF file content:")

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        # Use LangChain's ChatOpenAI wrapper for chat models
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7
        ) 

        # Set the llm_cache here - experiment
        set_llm_cache(InMemoryCache())
        # end of experiment

        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            st.write("execution time", timeit.timeit()) # cycle time experiment to be removed. For debugging only
            print(cb)
        st.write(response)
        st.write("cost", cb.total_cost)

if __name__ == "__main__":
    main()