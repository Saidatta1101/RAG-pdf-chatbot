import streamlit as st
from rag_pipeline import create_rag_pipeline, ask_question
import tempfile

st.title("📄 RAG PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    @st.cache_resource
    def load_pipeline(pdf_path):
        return create_rag_pipeline(pdf_path)

    if st.button("Process PDF"):
        with st.spinner("Processing..."):
            db, llm = load_pipeline(pdf_path)
            st.session_state.db = db
            st.session_state.llm = llm

    if "db" in st.session_state:
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("🤖 Thinking..."):
                answer = ask_question(
                    st.session_state.db,
                    st.session_state.llm,
                    query
                )

            st.write("### Answer:")
            st.write(answer)
