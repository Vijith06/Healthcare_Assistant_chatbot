from langchain_core.output_parsers import StrOutputParser
import models
import prompts
import vectordb


#### GENERATION ####
def generate_healthcare_chain(age,symptoms):
    """
    Generate Health using basic prompt LLM chain

    Args:
        age - age of the person
        symptoms- ex..fever

    Returns:
        response.content -> str
    """
        
    llm = models.create_chat_groq_model()

    prompt_template = prompts.healthcare_generator_prompt()

    chain = prompt_template | llm

    response = chain.invoke({
        "age" :age,
        "symptoms": symptoms
    })
    return response.content


#### RETRIEVAL and GENERATION ####

def generate_healthcare_rag_chain(age,symptoms, vector):
    """
    Creates a RAG chain for retrieval and generation.

    Args:
        age -> age of the person
        symptoms -> Ex. fever,cold 
        vectorstore ->  Instance of vector store 

    Returns:
        rag_chain -> rag chain
    """
    # Prompt
    prompt = prompts.healthcare_generator_rag_prompt()

    # LLM
    llm = models.create_chat_groq_model()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vectordb.retrieve_from_chroma(age,symptoms, vectorstore=vector)
    # Chain
    rag_chain = prompt| llm | StrOutputParser()

    response = rag_chain.invoke({
        "context" : format_docs(retriever),
        "age" :age,
        "symptoms":symptoms
    })    

    return response
