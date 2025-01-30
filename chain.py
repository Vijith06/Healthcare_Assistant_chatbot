from langchain_core.output_parsers import StrOutputParser
import model
import prompt
import vectordb


#### GENERATION ####
def generate_quiz_chain(level,field):
    """
    Generate quiz using basic prompt LLM chain

    Args:
        level - medium/easy/hard
        topic - topic for the quiz

    Returns:
        response.content -> str
    """
        
    llm = model.create_chat_groq_model()

    prompt_template = prompt.quiz_generator_prompt_from_hub()
    # prompt_template = prompts.quiz_generator_prompt_from_hub()

    chain = prompt_template | llm

    response = chain.invoke({
        "level" : level,
        "field" : field
    })
    
    return response.content


#### RETRIEVAL and GENERATION ####

def generate_quiz_rag_chain(level,field, vector):
    """
    Creates a RAG chain for retrieval and generation.

    Args:
        topic - topic for retrieval
        vectorstore ->  Instance of vector store 

    Returns:
        rag_chain -> rag chain
    """
    # Prompt
    prompt_template = prompt.quiz_generator_rag_prompt_from_hub()

    # LLM
    llm = model.create_chat_groq_model()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vectordb.retrieve_from_chroma(field, vectorstore=vector)
    # Chain
    rag_chain = prompt_template| llm | StrOutputParser()

    response = rag_chain.invoke({
        "context" : format_docs(retriever),
        "level":level,
        "field": field
    })    

    return response
