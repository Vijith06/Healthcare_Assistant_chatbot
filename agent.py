from langchain.agents import create_react_agent, AgentExecutor
import tool
import model
import prompt


#### AGENT ####
def generate_quiz_with_agent(level, field):
    """
    Generate a quiz using a LangChain agent.

    Args:
        level (str): Difficulty level (easy, medium, hard).
        field (str): Topic for the quiz.

    Returns:
        str: Generated quiz.
    """
    tools_list = [tool.quiz_generator_tool()]

    prompt_template = prompt.quiz_generator_agent()
    llm = model.create_chat_groq_model()
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=True,
        verbose=True,
        stop_sequence=True,
        max_iterations=3
    )

    # Ensure we pass a dictionary
    
    response = agent_executor.invoke({"level": level, "field": field})
    

    
    return response

#### AGENT WITH RAG ####
def generate_quiz_with_rag_agent(level, field, vector):
    """
    Generate a quiz using a LangChain agent with Retrieval-Augmented Generation (RAG).

    Args:
        level (str): Difficulty level of the quiz (e.g., "easy", "medium", "hard")
        field (str): Field or subject of the quiz (e.g., "math", "history", "science")
        vector (object): Instance of vector store

    Returns:
        str: Generated quiz
    """
    # Define tools for the agent
    tools_list = [
        tool.rag_retriever_tool(vector),
        tool.quiz_generator_tool()  # Tool to generate quizzes
    ]

    # Initialize the agent with the RAG-enabled prompt template
    prompt_template = prompt.quiz_generator_agent_with_rag()  # A new template for quiz generation
    llm = model.create_chat_groq_model()
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools_list, handle_parsing_errors=True, verbose=True, stop_sequence=True, max_iterations=3)

    # Agent interaction
    response = agent_executor.invoke({"input": f"Generate a {level} quiz on {field}"})
    return response
