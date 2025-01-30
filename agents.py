from langchain.agents import AgentExecutor, create_react_agent
from tools import healthcare_generator_tool, rag_retriever_tool
from prompts import healthcare_agent_prompt
from models import create_chat_groq_model
from vectordb import retrieve_from_chroma

def healthcare_chatbot_with_agent(symptom: str, age: int, vector_store):
    """
    A healthcare chatbot that provides food recommendations, dos and don'ts, and prescribes medicine based on symptoms and age.

    Args:
        symptom (str): The symptom described by the user.
        age (int): The age of the user.
        vector_store: The vector store instance for RAG retrieval.

    Returns:
        str: A response containing food recommendations, dos and don'ts, and prescribed medicine with a disclaimer.
    """
    # Define tools for the agent
    tools_list = [
        healthcare_generator_tool(),
        rag_retriever_tool(vector_store)
    ]

    # Initialize the agent with a healthcare-specific prompt template
    prompt_template = healthcare_agent_prompt()
    llm = create_chat_groq_model()  # Replace with your preferred LLM
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=True,
        verbose=True,
        stop_sequence=True,
        max_iterations=3
    )

    # Agent interaction
    response = agent_executor.invoke({"input": f"Symptom: {symptom}, Age: {age}"})
    
    # Add a disclaimer about consulting a doctor
    disclaimer = "\n\n**Disclaimer:** The prescribed medicine is a general recommendation. It is strongly advised to consult a healthcare professional before using any medication."
    return response + disclaimer


#Agent with rag
def healthcare_chatbot_with_rag_agent(symptom: str, age: int, vector_store):
    """
    A healthcare chatbot using a RAG (Retrieval-Augmented Generation) agent that provides food recommendations, dos and don'ts, and prescribes medicine based on symptoms and age.

    Args:
        symptom (str): The symptom described by the user.
        age (int): The age of the user.
        vector_store: The vector store instance for RAG retrieval.

    Returns:
        str: A response containing food recommendations, dos and don'ts, and prescribed medicine with a disclaimer.
    """
    # Define tools for the RAG agent
    tools_list = [
        healthcare_generator_tool(),
        rag_retriever_tool(vector_store)
    ]

    # Initialize the agent with a healthcare-specific prompt template
    prompt_template = healthcare_rag_agent_prompt()
    llm = create_chat_groq_model()  # Replace with your preferred LLM
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=True,
        verbose=True,
        stop_sequence=True,
        max_iterations=3
    )

    # Agent interaction
    response = agent_executor.invoke({"input": f"Symptom: {symptom}, Age: {age}"})
    
    # Add a disclaimer about consulting a doctor
    disclaimer = "\n\n**Disclaimer:** The prescribed medicine is a general recommendation. It is strongly advised to consult a healthcare professional before using any medication."
    return response + disclaimer
