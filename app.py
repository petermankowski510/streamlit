import streamlit as st
import streamlit_extras.add_vertical_space import add_vertical_space
"""Edit to test GIT"""

#Sidebar contents
with st.sitebar:
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