import streamlit as st
import chain
import vectordb
import agent

def quiz_generator_app():
    """
    Generates quiz Generator App with Streamlit, providing user input and displaying output.
    Includes a sidebar with two sections: Poem Generator and File Ingestion for RAG.
    """

    # Sidebar configuration
    st.sidebar.title("Menu")
    section = st.sidebar.radio(
        "Choose a section:",
        ("quiz Generator RAG", "RAG File Ingestion")
    )

    # db initialization
    vectordatabase = vectordb.initialize_chroma()

    # Condition for poem generation page
    if section == "quiz Generator RAG":
      st.title("Lets generate a quiz ! 👋")

      with st.form("quiz_generator"):
          levels=["Hard","Medium","Easy"]
          level = st.selectbox("📊 Select Difficulty Level for Quiz", levels)
          field = st.text_input(
            "Enter a topic for the quiz:"
          )
          submitted = st.form_submit_button("Submit")

          is_rag_enabled = st.checkbox("Check me to enable RAG")
          is_agent_enabled = st.checkbox("Check me to enable Agent")

          if submitted:
              if is_rag_enabled and is_agent_enabled:
                  response = agent.generate_quiz_with_rag_agent(level,field, vectordatabase)
              elif is_agent_enabled:
                  response = agent.generate_quiz_with_agent(level,field)
              elif is_rag_enabled:
                  response = chain.generate_quiz_rag_chain(level,field, vectordatabase)
              else:
                  response = chain.generate_quiz_chain(level,field)
             
              st.info(response)
    
    # Condition for RAG File Ingestion
    elif section == "RAG File Ingestion":
        st.title("RAG File Ingestion")

        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "docx", "pdf"])

        if uploaded_file is not None:
            vectordb.store_pdf_in_chroma(uploaded_file, vectordatabase)
            st.success(f"File '{uploaded_file.name}' uploaded  and file embedding stored in vectordb successfully!")

quiz_generator_app()
