import os
import requests
from bs4 import BeautifulSoup
import urllib.parse
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
# Removed: from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables from .env file
load_dotenv()

# --- Helper function for StackOverflow Search Tool ---
def search_stackoverflow_and_parse(query: str, num_results: int = 3) -> str:
    """
    Searches StackOverflow for a given query using requests and BeautifulSoup,
    and returns a formatted string of the top search results.
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        # StackOverflow search URL structure
        search_url = f"https://stackoverflow.com/search?q={encoded_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        print(f"Tool: Searching StackOverflow with URL: {search_url}")
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results_summary = []
        
        # Selector for search result items. This might need adjustment if StackOverflow's HTML changes.
        # Common class for search result containers: 'js-search-results' then 's-post-summary'
        search_items_container = soup.find('div', class_='js-search-results')
        if not search_items_container:
            # Fallback if the main container isn't found, try to find individual summaries directly
            search_items = soup.select('div.s-post-summary', limit=num_results)
            if not search_items: # Another fallback for older structures or different result pages
                 search_items = soup.select('div.search-result', limit=num_results)
        else:
            search_items = search_items_container.select('div.s-post-summary', limit=num_results)


        if not search_items:
            return "No search result items found on the page. StackOverflow's structure might have changed or the query yielded no results."

        for item in search_items:
            title_tag = item.select_one('h3.s-post-summary--content-title a.s-link')
            if not title_tag: # Fallback selector for title
                title_tag = item.select_one('div.result-link a') # Common in older SO layouts

            if title_tag:
                title = title_tag.get_text(strip=True)
                link = title_tag.get('href')
                if link and not link.startswith('http'):
                    link = f"https://stackoverflow.com{link}"
                
                # Try to get an excerpt/snippet
                excerpt_tag = item.select_one('div.s-post-summary--content-excerpt')
                snippet = excerpt_tag.get_text(strip=True) if excerpt_tag else "No snippet available."
                
                # Try to get votes/answers count if available (example)
                stats_tag = item.select_one('div.s-post-summary--stats-item.s-post-summary--stats-item__emphasized')
                votes_or_answers = stats_tag.get_text(strip=True) if stats_tag else "Stats N/A"

                results_summary.append(f"Title: {title}\nLink: {link}\nStats: {votes_or_answers}\nSnippet: {snippet}\n---")

        if not results_summary:
            return "No relevant StackOverflow results found or parsed for your query."
            
        return "\n".join(results_summary)

    except requests.exceptions.Timeout:
        return "Error: The request to StackOverflow timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP error occurred while searching StackOverflow: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: A network problem occurred during StackOverflow search: {e}"
    except Exception as e:
        # Catch any other exceptions during parsing
        return f"Error: An unexpected error occurred while parsing StackOverflow results: {e}"


def create_stackoverflow_coding_agent():
    """
    Creates a LangChain agent capable of answering coding questions by searching
    StackOverflow using a custom tool (Requests + BeautifulSoup) and a Groq language model.
    """
    # 1. Define the tools the agent can use
    # We'll use a custom tool that uses requests and BeautifulSoup to search StackOverflow.
    stackoverflow_search_tool = Tool(
            name="StackOverflow_Search_and_Parse", # Name used in the prompt
            func=search_stackoverflow_and_parse,
            description="Useful for when you need to answer coding questions or find solutions to programming problems. Input should be a search query string for StackOverflow."
        )
    tools = [stackoverflow_search_tool]

    # 2. Define the LLM to use
    # This uses ChatGroq. Ensure GROQ_API_KEY is set in your environment.
    try:
        # Get API key from .env file
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
            
        # We are using Llama 3 via Groq.
        # Temperature is set lower for more factual coding answers.
        llm = ChatGroq(temperature=0.1, model_name="llama3-8b-8192")
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        print("Please ensure the GROQ_API_KEY is properly set in your .env file.")
        return None

    # 3. Define the prompt template with correct formatting
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful coding assistant. Your primary goal is to answer coding questions by finding relevant information on StackOverflow.
                     You have access to the following tools:
                     {tools}
                     
                     Available tools: {tool_names}
                     
                     To answer a coding question, follow these steps:
                     1.  **Understand the Question**: Clearly identify the core problem or information needed.
                     2.  **Formulate Search Query**: Create a concise and effective search query for StackOverflow.
                     3.  **Use the Tool**:
                         Thought: I need to search StackOverflow for [your reasoning and specific query].
                         Action: StackOverflow_Search_and_Parse
                         Action Input: [your search query for StackOverflow]
                     4.  **Analyze Results**:
                         Observation: [The tool will return a summary of search results, including titles, links, snippets, and sometimes stats. It might also return an error message.]
                         Carefully review the observed results. If an error occurred or no results were found, state that and consider rephrasing the query.
                     5.  **Refine if Necessary**: If the initial results are not satisfactory (e.g., not relevant, error from tool), think about how to improve the search query and repeat step 3 and 4.
                     6.  **Formulate Answer**:
                         Thought: I have found relevant information on StackOverflow (or determined that no sufficient information was found). I will now synthesize an answer.
                         Final Answer: Provide a clear, concise answer to the user's coding question.
                         - If relevant information is found, summarize it.
                         - If possible, include relevant code snippets based on the StackOverflow information.
                         - Mention the source (e.g., "According to a StackOverflow post titled '...'").
                         - You can include direct links to the most helpful StackOverflow pages if they provide more depth.
                         - If no good answer is found after searching, clearly state that you couldn't find a definitive answer on StackOverflow.
                         - If the tool returns an error, report the error and explain that you couldn't complete the search."""),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}") # agent_scratchpad is a placeholder for the agent's intermediate steps
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
        handle_parsing_errors=True, # Handles errors if the LLM output is not in the expected ReAct format
        max_iterations=4 # Adjust as needed; 3-5 is usually good for ReAct
    )

    return agent_executor

# --- Example Usage ---
if __name__ == '__main__':
    # This block runs when the script is executed directly.
    print("Attempting to create and run the StackOverflow coding question agent with Groq...")

    # Create the agent executor instance
    agent_executor = create_stackoverflow_coding_agent()

    if agent_executor:
        # Define the query for the agent (a coding question)
        # query = "How to sort a list of dictionaries by a specific key in Python?"
        # query = "python requests get timeout"
        query = "How to handle 'ModuleNotFoundError' in Python when importing local files?"


        print(f"\nRunning agent with query: '{query}'")

        try:
            # Invoke the agent with the query
            # The agent will use its tools (StackOverflow_Search_and_Parse) and the Groq LLM to respond.
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