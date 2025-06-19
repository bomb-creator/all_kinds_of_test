import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables from .env file
load_dotenv()

def create_ai_news_tweet_agent():
    """
    Creates a LangChain agent capable of generating daily tweet ideas for AI news
    using a search tool and a Groq language model.
    """
    # 1. Define the tools the agent can use
    # We'll use DuckDuckGo Search as a general tool to find AI news for tweet ideas.
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for when you need to find recent AI news or information to generate daily tweet ideas."
        )
    ]

    # 2. Define the LLM to use
    # This now uses ChatGroq. Ensure GROQ_API_KEY is set in your environment.
    try:
        # Get API key from .env file
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
            
        # We are using Llama 3 via Groq, which is known for its speed.
        # You can explore other models available on the Groq platform.Or use open ai API or anything in ur choice 
        #I havent work with google llm with agent specelly thise langchaing feel free to explore that possibalty also 
        llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192") # Adjusted temperature for more creative ideas
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        print("Please ensure the GROQ_API_KEY is properly set in your .env file.")
        return None

    # 3. Define the prompt template with correct formatting
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that generates daily tweet ideas for AI news.
                     You should aim for engaging, concise, and informative tweet ideas.
                     You have access to the following tools:
                     {tools}
                     
                     Available tools: {tool_names}
                     
                     To use a tool, please use the following format:
                     Thought: I need to use a tool to find recent AI news or inspiration for tweet ideas.
                     Action: tool_name
                     Action Input: input for the tool (e.g., "latest AI breakthroughs", "AI ethics news")
                     Observation: tool output
                     ... (repeat this Thought/Action/Observation pattern as needed)
                     Thought: I now have enough information to generate tweet ideas.
                     Final Answer: your list of tweet ideas, formatted clearly. For example:
                     1. Tweet Idea: [Idea 1] #AI #TechNews
                     2. Tweet Idea: [Idea 2] #FutureOfAI #Innovation
                     3. Tweet Idea: [Idea 3] #MachineLearning #AIUpdate"""),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    # 4. Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template
    )

    # 5. Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5  # Increased max_iterations for potentially more complex idea generation
    )

    return agent_executor

# --- Example Usage ---
if __name__ == '__main__':
    # This block runs when the script is executed directly.
    print("Attempting to create and run the AI news tweet ideas agent with Groq...")

    # Create the agent executor instance
    agent_executor = create_ai_news_tweet_agent()

    if agent_executor:
        # Define the query for the agent
        query = "Generate 3 daily tweet ideas about recent AI news. Focus on breakthroughs or interesting applications."

        print(f"\nRunning agent with query: '{query}'")

        try:
            # Invoke the agent with the query
            # The agent will use its tools and the Groq LLM to respond.
            result = agent_executor.invoke({'input': query})

            # Print the final output from the agent
            print("\n--- Agent Final Output ---")
            print(result['output'])
            print("--------------------------")

        except Exception as e:
            print(f"\nAn error occurred during agent execution: {e}")
            print("Please check the error message and ensure your environment is set up correctly.")
    else:
        print("Agent creation failed. Please check the error messages above.")

    print("\nScript finished.")