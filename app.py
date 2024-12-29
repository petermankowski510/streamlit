#typo in sidebar
#https://www.youtube.com/watch?v=RIWbalZ7sTo

import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""Edit to test GIT"""

#Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)streamlit run app.py
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with by [Prompt Engineer](http://youtube.com/@engineeringprompt)')

def main():
    st.header("Chat with PDF")

    #Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    #st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    st.write(chunks)

        # st.write(text)

if __name__== '__main__':
    main()