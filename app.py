import streamlit as st
import chains
import vectordb
import agents

def healthcare_generator_app():
    """
    Generates healthcare assistant  App with Streamlit, providing user input and displaying output.
    Includes a sidebar with two sections: Poem Generator and File Ingestion for RAG.
    """

    # Sidebar configuration
    st.sidebar.title("Menu")
    section = st.sidebar.radio(
        "Choose a section:",
        ("HealthCare Assistant RAG", "RAG File Ingestion")
    )

    # db initialization
    vectordatabase = vectordb.initialize_chroma()

    # Condition for poem generation page
    if section == "HealthCare Assistant RAG":
      st.title("Lets Assist the Paitent ! 👋")

      with st.form("healthcare_generator"):
          age = st.text_input(
            "Enter a your age:"
          )
          symptoms = st.text_input(
              "Enter the symptoms Eg..fever,cold .."
          )
          submitted = st.form_submit_button("Submit")

          is_rag_enabled = st.checkbox("Check me to enable RAG")
          is_agent_enabled = st.checkbox("Check me to enable Agent")

          if submitted:
              if is_rag_enabled and is_agent_enabled:
                  response = agents.generate_healthcare_with_rag_agent(age,symptoms, vectordatabase)
              elif is_agent_enabled:
                  response = agents.generate_healthcare_with_agent(age,symptoms)
              elif is_rag_enabled:
                  response = chains.generate_healthcare_rag_chain(age,symptoms, vectordatabase)
              else:
                  response = chains.generate_healthcare_chain(age,symptoms)
             
              st.info(response)
    
    # Condition for RAG File Ingestion
    elif section == "RAG File Ingestion":
        st.title("RAG File Ingestion")

        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "docx", "pdf"])

        if uploaded_file is not None:
            vectordb.store_pdf_in_chroma(uploaded_file, vectordatabase)
            st.success(f"File '{uploaded_file.name}' uploaded  and file embedding stored in vectordb successfully!")

healthcare_generator_app()
