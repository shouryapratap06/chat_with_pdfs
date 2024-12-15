# IMPORTS
import streamlit as st
import pickle
from dotenv import load_dotenv

# For pdf functionality
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI  # will help in using LLMs
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import os


# sidebar contents
with st.sidebar:
    st.title(" 📒😁 Chat-with-PDF")
    st.markdown(
        """
        ## About
        Now you can chat with your PDF files.

        This app is Powered by: 
        - [Streamlit](https://streamlit.io/).
        - [LangChain](https://www.langchain.com/).
        - [OpenAI](https://openai.com/).

    """
    )

    add_vertical_space(5)
    st.write("Made with 🥰")


def main():
    st.header("Chat with PDF 📒😁")
    load_dotenv()
    # Upload a pdf file
    pdf = st.file_uploader("Upload a PDF file", type="pdf")
    # st.write(pdf.name)

    # Display the pdf file if it is uploaded
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader) - it will write the object name on the web
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text) - Will show the text of pdf on the web

        # Split the text into chunks of 1000 characters(token)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # st.write(chunks)

        # Embeddings - object below and will use the FAISS `VectorStore` as our Database

        # Writing files to storage
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            # Read file from the storage.
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        # st.write("🎰Embedding loaded from the disk")
        else:  # Recompute the embeddings
            embeddings = OpenAIEmbeddings()
            # Variable VectoreStore
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

            # st.write("🎰Embedding computation complete")

        # User Questions with AI Begun here
        query = st.text_input("Ask me anything about your PDF file")
        # st.write("You: ", query)

        if query:
            # LLM context window
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chian = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chian.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

            # st.write(docs)


if __name__ == "__main__":
    main()
