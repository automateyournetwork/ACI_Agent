import os
import json
import requests
import difflib
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ACIController for APIC Authentication and Data Fetching
class ACIController:
    def __init__(self, aci_url, username, password):
        self.aci = aci_url.rstrip('/')
        self.username = username
        self.password = password
        self.cookie = self.get_token()

    def get_token(self):
        url = f"{self.aci}/api/aaaLogin.json"
        payload = {
            "aaaUser": {
                "attributes": {
                    "name": self.username,
                    "pwd": self.password,
                }
            }
        }
        response = requests.post(url, json=payload, verify=False)
        response.raise_for_status()
        print(f"<Authentication Status code {response.status_code} for {url}>")
        return response.cookies

    def get_api(self, api_url: str, page: int = 0, page_size: int = 100):
        params = {"page": page, "page-size": page_size}
        response = requests.get(
            f"{self.aci}{api_url}",
            params=params,
            cookies=self.cookie,
            verify=False
        )
        response.raise_for_status()
        response_dict = response.json()
        total_count = int(response_dict.get("totalCount", 0))
        imdata = response_dict.get("imdata", [])

        # Retrieve more pages if needed
        while len(imdata) < total_count:
            page += 1
            params["page"] = page
            response = requests.get(
                f"{self.aci}{api_url}",
                params=params,
                cookies=self.cookie,
                verify=False
            )
            response.raise_for_status()
            imdata.extend(response.json().get("imdata", []))

        return {"totalCount": total_count, "imdata": imdata}

# Function to load supported URLs with their names from a JSON file
def load_urls(file_path='urls.json'):
    if not os.path.exists(file_path):
        return {"error": f"URLs file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Extract the URL and Name fields into a list of tuples
        url_list = [(entry['URL'], entry.get('Name', '')) for entry in data]
        return url_list
    except Exception as e:
        return {"error": f"Error loading URLs: {str(e)}"}

def check_url_support(api_url: str) -> dict:
    url_list = load_urls()
    if "error" in url_list:
        return url_list  # Return error if loading URLs failed

    # Separate URLs and Names into two lists for matching
    urls = [entry[0] for entry in url_list]
    names = [entry[1] for entry in url_list]

    # Find the closest matches to the input based on both URL and Name
    close_url_matches = difflib.get_close_matches(api_url, urls, n=1, cutoff=0.6)
    close_name_matches = difflib.get_close_matches(api_url, names, n=1, cutoff=0.6)

    # Determine the best match from either the URL or Name
    if close_url_matches:
        closest_url = close_url_matches[0]
        # Find the matching name for this URL
        matching_name = [entry[1] for entry in url_list if entry[0] == closest_url][0]
        return {"status": "supported", "closest_url": closest_url, "closest_name": matching_name}
    elif close_name_matches:
        closest_name = close_name_matches[0]
        # Find the corresponding URL for this name
        closest_url = [entry[0] for entry in url_list if entry[1] == closest_name][0]
        return {"status": "supported", "closest_url": closest_url, "closest_name": closest_name}
    else:
        return {"status": "unsupported", "message": f"The input '{api_url}' is not supported. Please check the available URLs or Names."}
    
@tool
def check_supported_url_tool(api_url: str) -> dict:
    """Check if an API URL or Name is supported by the ACI controller."""
    result = check_url_support(api_url)
    if result.get('status') == 'supported':
        # Automatically get the data if the URL is valid
        closest_url = result['closest_url']
        closest_name = result['closest_name']
        return {
            "status": "supported",
            "message": f"The closest supported API URL is '{closest_url}' ({closest_name}).",
            "action": {
                "next_tool": "get_aci_data_tool",
                "input": closest_url
            }
        }
    return result

@tool
def get_aci_data_tool(api_url: str) -> dict:
    """Fetch data from the ACI controller."""
    try:
        aci_controller = ACIController(aci_url="https://10.10.20.14", username="admin", password="C1sco12345")
        data = aci_controller.get_api(api_url)
        return data
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from ACI: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def process_agent_response(response):
    if response and response.get("status") == "supported" and "next_tool" in response.get("action", {}):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        # Automatically invoke the next tool (get_aci_data_tool)
        return agent_executor.invoke({
            "input": tool_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": "",
            "tool": next_tool
        })
    else:
        return response

# Create a list of tools
tools = [check_supported_url_tool, get_aci_data_tool]

# Render text descriptions for the tools
tool_descriptions = render_text_description(tools)

# Create the PromptTemplate
template = """
Assistant is a network assistant with the capability to gather data from Cisco ACI controllers.

NETWORK INSTRUCTIONS:

Assistant is designed to retrieve information from the Cisco ACI controller using provided tools. You MUST use these tools for checking available data and fetching it.

Assistant has access to a list of API URLs and their associated Names provided in a 'urls.json' file. You can use the 'Name' field to find the appropriate API URL to use.

**Important Guidelines:**

1. **If you are certain of the API URL or the Name of the data you want, use the 'get_aci_data_tool' to fetch data.**
2. **If you are unsure of the API URL or Name, or if there is ambiguity, use the 'check_supported_url_tool' to verify the URL or Name or get a list of available ones.**
3. **If the 'check_supported_url_tool' finds a valid URL or Name, automatically use 'get_aci_data_tool' to fetch data from that URL.**
4. **Do NOT use any unsupported URLs or Names.**

**Using the Tools:**

- If you are confident about the API URL or Name, use the 'get_aci_data_tool'.
- If there is any doubt or ambiguity, always check the URL or Name first with the 'check_supported_url_tool'.

To use a tool, follow this format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

If the first tool provides a valid URL or Name, you MUST immediately run the 'get_aci_data_tool' without waiting for another input. Follow the flow like this:

**Example:**

Thought: Do I need to use a tool? Yes
Action: check_supported_url_tool
Action Input: "Leaf Nodes"
Observation: "The closest supported API URL is '/api/node/class/topSystem.json' (Leaf Nodes)."

Thought: Do I need to use a tool? Yes
Action: get_aci_data_tool
Action Input: "/api/node/class/topSystem.json"
Observation: [retrieved data here]

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

**Correct Formatting is Essential:** Ensure that every response follows the format strictly to avoid errors.

TOOLS:

Assistant has access to the following tools:

- check_supported_url_tool: Checks if an API URL or Name is supported by the ACI controller.
- get_aci_data_tool: Fetches data from the ACI controller using the specified API URL.

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
"""

# Define input variables
input_variables = ["input", "agent_scratchpad"]

# Create the PromptTemplate
prompt_template = PromptTemplate(
  template=template,
  input_variables=input_variables,
  partial_variables={
      "tools": tool_descriptions,
      "tool_names": ", ".join([t.name for t in tools])
  }
)

# Initialize the LLM (you can replace 'gpt-3.5-turbo' with your desired model)
#llm = Ollama(model="llama3.1", temperature=0)
llm = ChatOpenAI(model_name="gpt-4o")

# Create the ReAct agent
agent = create_react_agent(
  llm=llm,
  tools=tools,
  prompt=prompt_template
)

# Create the AgentExecutor
agent_executor = AgentExecutor(
  agent=agent,
  tools=tools,
  handle_parsing_errors=True,
  verbose=True,
  max_iterations=10
)
# ============================================================
# Streamlit App
# ============================================================

# Initialize Streamlit
st.title("ACI AI Agent with LangChain")
st.write("Ask your network questions and get insights using AI!")

# Input for user questions
user_input = st.text_input("Enter your question:")

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Button to submit the question
if st.button("Send"):
    if user_input:
        # Add the user input to the conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})

        # Invoke the agent with the user input and current chat history
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history,
                "agent_scratchpad": ""  # Initialize agent scratchpad as an empty string
            })

            # Check if chaining is needed (i.e., next tool)
            final_response = process_agent_response(response)

            # Extract the final answer
            final_answer = final_response.get('output', 'No answer provided.')

            # Display the question and answer
            st.write(f"**Question:** {user_input}")
            st.write(f"**Answer:** {final_answer}")

            # Add the response to the conversation history
            st.session_state.conversation.append({"role": "assistant", "content": final_answer})

            # Update chat history with the new conversation
            st.session_state.chat_history = "\n".join(
                [f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation]
            )
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")

# Display the entire conversation history
if st.session_state.conversation:
    st.write("## Conversation History")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")
