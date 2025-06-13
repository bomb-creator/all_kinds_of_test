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

# Ensure the API key is set (you can uncomment and set it here for testing)
# if 'GROQ_API_KEY' not in os.environ:
#     os.environ['GROQ_API_KEY'] = 'gsk_YOUR_KEY_HERE' # Replace with your actual key if testing this way

def create_weather_aq_agent():
    """
    Creates a LangChain agent capable of finding current weather temperature
    and air quality information using a search tool and a Groq language model.
    """
    # 1. Define the tools the agent can use
    # We'll use DuckDuckGo Search as a general tool.
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for when you need to answer questions about current weather conditions (like temperature, humidity, wind) and air quality (like AQI - Air Quality Index) for a specific location. Be specific with your search query, e.g., 'current temperature in London' or 'AQI in Beijing'."
        )
    ]

    # 2. Define the LLM to use
    # This uses ChatGroq. Ensure GROQ_API_KEY is set in your environment.
    try:
        # We are using Llama 3 via Groq, known for its speed.
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        print("Please ensure the GROQ_API_KEY environment variable is set correctly.")
        print("You can get a key from: https://console.groq.com/keys")
        return None

    # 3. Define the prompt template with correct formatting
    # Note: The {agent_scratchpad} placeholder is crucial for the ReAct agent.
    # It will be populated with the sequence of thoughts, actions, and observations.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can find current weather temperature and air quality information for specified locations.
You have access to the following tools:
{tools}

To use a tool, you MUST use the following format:

Thought: Do I need to use a tool? Yes. I need to find [specific information] for [location].
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action, e.g., "current temperature in Paris" or "air quality index in New Delhi".
Observation: The result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: Do I need to use a tool? No. I have all the information I need.
Final Answer: Your comprehensive answer to the human's original question, summarizing the weather and/or air quality. If providing temperature, try to give it in both Celsius and Fahrenheit if easily found. For AQI, mention the value and its general meaning (e.g., Good, Moderate, Unhealthy)."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}") # Crucial for ReAct agent
    ])


    # 4. Create the agent
    # This uses the create_react_agent function.
    try:
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template
        )
    except Exception as e:
        print(f"Error creating ReAct agent: {e}")
        return None

    # 5. Create the agent executor
    # This will run the agent and manage the interaction loop.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Set to True to see the agent's thought process
        handle_parsing_errors="Check your output and make sure it conforms to the ReAct format.", # Provides guidance on parsing errors
        max_iterations=5, # Increased slightly for potentially two searches (weather + AQI)
        # early_stopping_method="generate" # Can be useful if the LLM generates a Final Answer early
    )

    return agent_executor

# --- Example Usage ---
if __name__ == '__main__':
    print("Attempting to create and run the weather and air quality agent with Groq...")

    # Check for API key before proceeding
    if 'GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY'].startswith('gsk_'):
        print("\nERROR: GROQ_API_KEY environment variable is not set or is invalid.")
        print("Please set it before running the script.")
        print("For macOS/Linux: export GROQ_API_KEY='your-api-key'")
        print("For Windows: set GROQ_API_KEY=your-api-key")
        print("You can get a key from: https://console.groq.com/keys")
        exit()
    else:
        print(f"GROQ_API_KEY found (partially): {os.environ['GROQ_API_KEY'][:8]}...")


    # Create the agent executor instance
    weather_agent_executor = create_weather_aq_agent()

    if weather_agent_executor:
        # Define queries for the agent
        queries = [
            "What is the current temperature and air quality in London, UK?",
            "How's the weather and AQI in Tokyo, Japan today?",
            "Can you tell me the temperature in Celsius and Fahrenheit, and the air quality index for New York City?"
        ]

        for query in queries:
            print(f"\nRunning agent with query: '{query}'")
            try:
                # Invoke the agent with the query
                result = weather_agent_executor.invoke({'input': query})

                # Print the final output from the agent
                print("\n--- Agent Final Output ---")
                print(result['output'])
                print("--------------------------")

            except Exception as e:
                print(f"\nAn error occurred during agent execution for query '{query}': {e}")
                print("Please check the error message and ensure your environment (especially GROQ_API_KEY) is set up correctly.")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
    else:
        print("Weather and Air Quality Agent creation failed. Please check the error messages above.")

    print("\nScript finished.")

""""got whuge ammoubnt of errors need to debut alteast come to known that is thas some attiribute erroe and 
some modre i thing vaiable bugs so need to fixx thosse 6.13.2024"""