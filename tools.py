from langchain.agents import Tool
import chains
from vectordb import retrieve_from_chroma

def healthcare_generator_tool():
    """
    Generate a tool that can create healthcare bot

    Args:
        age -> age of the person
        symptoms -> like fever,cold

    Returns:
        healthcare assistant generator Tool 
    """
    return Tool(
            name="healthcare generator",
            func=lambda age,symptoms: chains.generate_healthcare_chain(age,symptoms),
            description="Provides medical support to patients with the symptoms."
        )

def rag_retriever_tool(vector):
    """
    Create a Tool for retrieving relevant documents using RAG

    Args:
        vector (object): The vector store instance.

    Returns:
        Tool: A LangChain Tool object for RAG retrieval.
    """
    return Tool(
            name="RAG Retriever",
            func=lambda age,symptoms: "\n\n".join(
                doc.page_content for doc in retrieve_from_chroma(age,symptoms, vectorstore=vector)
            ),
            description="Retrieves relevant documents for a given topic using a vector store."
        )
