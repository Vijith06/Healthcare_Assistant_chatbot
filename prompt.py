from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.prompts import PromptTemplate



def quiz_generator_prompt_from_hub():
    """
    Generates Prompt template from the LangSmith prompt hub
    Returns:
        ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
    """
    prompt_template = hub.pull("vijith/quiz_generator")
    return prompt_template


def quiz_generator_rag_prompt():
    """
    Generates a RAG-enabled Prompt template for quiz generation.

    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """

    system_msg = '''
                You are a dedicated quiz generator assistant, specialized in crafting quizzes tailored to the user's requested level and field. Your task is strictly to generate quizzes based on the provided parameters. Follow these guidelines:

                1. Only respond to queries explicitly requesting a quiz for a specific level and field.
                2. The output must strictly consist of a list of questions, formatted as follows:
                   - For multiple-choice questions: Include the question, four options (A, B, C, D), and the correct answer.
                   - For true/false questions: Include the statement and the correct answer (True or False).
                   - For open-ended questions: Include only the question.
                3. Do not include any additional explanations, descriptions, or headers beyond the questions themselves.
                4. If the query is unrelated to quiz generation (e.g., generating poems, recipes, suggestions, or general knowledge questions), respond with:
                   "I am a quiz generator assistant, expert in generating quizzes based on the provided level and field. Please ask me a quiz-related query."
                5. Do not perform any tasks beyond quiz generation. Always fall back to the above message for non-quiz-related queries.

                Note: Ensure that the generated quiz aligns with the specified level (e.g., beginner, intermediate, advanced) and field (e.g., Math, Science, History). Incorporate relevant context from external sources if provided in the conversation. The quizzes must reflect the nuances of the requested level and field.
                '''

    user_msg = "Generate a quiz for the {field} field at the {level} level."

    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])

    return prompt_template



def quiz_generator_rag_prompt_from_hub(template="vijith/quiz_rag_generator"):
    """
    Generates Prompt template from the LangSmith prompt hub

    Returns:
        ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
    """
    
    prompt_template = hub.pull(template)
    return prompt_template






def quiz_generator_agent():
    """
    Creates a prompt template for an agent to generate quizzes based on a given topic and difficulty level.

    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """
    system_msg = '''
            You are a highly skilled quiz generator agent, specialized in crafting quizzes based on specific topics and difficulty levels. Answer the following questions as best you can. You have access to the following tools:
            {tools}

            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do with the following restrictions:
            1. Only generate quizzes explicitly requested with a given field (topic) and difficulty level (easy, medium, hard).
            2. Ensure that questions are well-structured, relevant to the topic, and aligned with the given difficulty level.
            3. If the query is unrelated to quiz generation (e.g., generating code, poems, recipes, suggestions, general knowledge questions, or any other non-quiz tasks), respond with:
               "I am a quiz generator agent, expert in generating quizzes based on specific topics and difficulty levels. Please ask me a quiz-related query."
            4. Do not perform any tasks beyond quiz generation. Always fall back to the above message for non-quiz-related queries.

            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat for a maximum of N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!
            
            Question: Generate a quiz on the topic "{{field}}" with difficulty level "{{level}}".
            Thought: {agent_scratchpad}
    '''

    user_msg = "Generate a quiz for the {field} field at the {level} level."

    # Create a ChatPromptTemplate and pass the system and user messages
    prompt_template = ChatPromptTemplate(
        messages=[
            ("system", system_msg),
            ("user", user_msg)
        ],
        input_variables=["level", "field", "tools", "agent_scratchpad"]  # Ensure these variables are included
    )

    return prompt_template


# def poem_generator_agent_from_hub(template="poem-generator/agent"):
#     """
#     Generates an template for agent to generate poem from the LangSmith hub.

#     Returns:
#         ChatPromptTemplate -> ChatPromptTemplate pulled from LangSmith Hub
#     """
#     agent = hub.pull(template, object_type="agent")
#     return agent






def quiz_generator_agent_with_rag():
    """
    Creates an agent with RAG capabilities for generating quizzes.

    Returns:
        PromptTemplate -> Configured PromptTemplate instance
    """
    prompt_template = '''
            You are a dedicated quiz generator agent, specialized in crafting quizzes based on difficulty levels and fields of study. Answer the following questions as best you can. You have access to the following tools:
            {tools}
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do with the following restrictions:
            1. Only respond to queries explicitly requesting a quiz based on level and field.
            2. The output must strictly be a quiz with questions relevant to the specified level and field, with no additional explanations, descriptions, or headers.
            3. If the query is related to quiz generation, use the RAG retriever tool first and use the context to generate the quiz using the quiz generation tool.
            4. If the query is unrelated to quiz generation (e.g., generating code, recipes, suggestions, general knowledge questions, or any other non-quiz tasks), respond with:
            "I am a quiz generator agent, expert in generating quizzes based on difficulty level and field. Please ask me a quiz-related query."
            5. Do not perform any tasks beyond quiz generation. Always fall back to the above message for non-quiz-related queries.
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat for maximum of N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            Begin!
            Question: {input}
            Thought: {agent_scratchpad}
            '''
    prompt = PromptTemplate(
        input_variables=["input", "tool_names", "agent_scratchpad"],
        template=prompt_template
    )
    return prompt

# def poem_generator_agent_with_rag_from_hub(template="poem-generator/agent-with-rag"):
#     """
#     Generates an agent with RAG capabilities for poem generation from the LangSmith hub.

#     Returns:
#         ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
#     """
#     agent = hub.pull(template, object_type="agent")
#     return agent
