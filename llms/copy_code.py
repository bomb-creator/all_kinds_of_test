import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# --- Configuration ---
# Set your Groq API key as an environment variable
# For macOS/Linux: export GROQ_API_KEY='your-api-key'
# For Windows: set GROQ_API_KEY=your-api-key
#
# Or, uncomment the line below and replace with your key (not recommended for production)
# os.environ['GROQ_API_KEY'] = 'YOUR_GROQ_API_KEY_HERE'

def create_weather_air_quality_agent():
    """
    Creates a LangChain agent capable of finding current weather and air quality
    information for locations around the world using a search tool and a Groq language model.
    """
    # 1. Define the tools the agent can use
    # We'll use DuckDuckGo Search as a general tool to find current information.
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for when you need to answer questions about current weather conditions, forecasts, or air quality in various locations around the world."
        )
    ]

    # 2. Define the LLM to use
    # This now uses ChatGroq. Ensure GROQ_API_KEY is set in your environment.
    try:
        # We are using Llama 3 via Groq, which is known for its speed.
        # You can explore other models available on the Groq platform.
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        print("Please ensure the GROQ_API_KEY environment variable is set correctly.")
        return None

    # 3. Define the prompt template with correct formatting
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can find current weather conditions, forecasts, and air quality information for locations around the world.
                     You have access to the following tools:
                     {tools}
                     
                     Available tools: {tool_names}
                     
                     To use a tool, please use the following format:
                     Thought: I need to use a tool to help with this
                     Action: tool_name
                     Action Input: input for the tool
                     Observation: tool output
                     ... (repeat this Thought/Action/Observation pattern as needed)
                     Thought: I now know the answer
                     Final Answer: your response to the human"""),
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
        max_iterations=5  # Adjusted for potentially more complex search/parse for weather & AQI
    )

    return agent_executor

# --- Example Usage ---
if __name__ == '__main__':
    # This block runs when the script is executed directly.
    print("Attempting to create and run the weather and air quality agent with Groq...")

    # Create the agent executor instance
    agent_executor = create_weather_air_quality_agent()

    if agent_executor:
        # Define the query for the agent
        query = "What is the current weather in London, UK, and the air quality in Beijing, China? Also, what's the forecast for Paris for tomorrow?"

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