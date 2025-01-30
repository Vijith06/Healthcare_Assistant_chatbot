from langchain.agents import Tool
import chain
from vectordb import retrieve_from_chroma

import json

def quiz_generator_tool():
    """
    Generate a tool that can create quizzes.

    Returns:
        Tool: Quiz generator tool.
    """
    return Tool(
        name="Quiz Generator",
        func=lambda inputs: (
    
         chain.generate_quiz_chain(**(json.loads(inputs) if isinstance(inputs, str) else inputs))
),


        description="Generates a quiz based on a given topic and difficulty level.",
    )


def rag_retriever_tool(vector):
    """
    Create a Tool for retrieving relevant documents using RAG.

    Args:
        vector (object): The vector store instance.

    Returns:
        Tool: A LangChain Tool object for RAG retrieval.
    """
    return Tool(
        name="RAG Retriever",
        func=lambda inputs: "\n\n".join(
            doc.page_content for doc in retrieve_from_chroma(
                inputs['level'], inputs['field'], vectorstore=vector
            )
        ) if isinstance(inputs, dict) else "\n\nInvalid inputs",  # Add a check for valid inputs
        description="Retrieves relevant documents for a given topic using a vector store."
    )
