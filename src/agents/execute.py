from langchain.tools import tool
# from nba_tools import NBATools
from semantic_router import RouteLayer, Route
from semantic_router.encoders import FastEmbedEncoder

# Initialize the tools and client
# browser_tools_instance = BrowserTools()
import os
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import (
    # DirectoryReadTool,
    # FileReadTool,
    SerperDevTool,
    # WebsiteSearchTool
)


class CLITool:
    @tool("Executor")
    def execute_cli_command(code_string: str):
        """Create and Execute code using Open Interpreter."""
        exec(code_string)
        # return result


load_dotenv()

PORTKEY_API_KEY_OPENAI = os.getenv('PORTKEY_KEY_OPENAI')
VIRTUAL_KEY_OPENAI = os.getenv('PORTKEY_VIRTUAL_KEY_OPENAI')

# Working Good
llm = ChatOpenAI(
    api_key='dummy',
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        provider="openai",
        api_key=PORTKEY_API_KEY_OPENAI,
        virtual_key=VIRTUAL_KEY_OPENAI
    )
)


# Define the mathematical Agent
class MathematicsExpert(Agent):
    def __init__(self):
        super().__init__(
            role='Mathematics Expert',
            goal="Break down the user maths problem into easy problems, solve them and give the output.",
            backstory="""
            You are a mathematics expert, who can solve the problem. You can solve any problem with breaking down into simpler problem and then solve it.
            You think step by step before solving any problem
            
            Example: 
            What is the cube of root of two. 
            
            Simpler tasks: 
            Calculate the root of two
            Solution: 1.4142 
            Calculate the cube of this number 
            Solution: 2.8284
            
            """,
            verbose=True,
            allow_delegation=False,
            # tools=[],
            llm=llm
        )


# Define the Coder Agent
class Coder(Agent):
    def __init__(self):
        super().__init__(
            role='Python Coder',
            goal="Generate the python code using PyAutoGUI for executing the request from user side. Only using keyboards based commmands of PyAutoGUI to complete the request",
            backstory="""
                Expert at breaking down a request for windows system into simpler steps based steps using keyboard shortcuts, then writing the code for commands in PyAutoGUI. I only use keyboard commands to run the given query. Here's how I tackle these specific queries:
              - Break down the user request into smaller steps and most easiest flows.
              - Identify which keyboards keys are required to execute the workflow of the smaller step. To locate a folder/file always use file explorer search bar - Win + E and then Control + F.
              - Write a python code for the given identified keyboards key using PyAutoGUI with 2 second delay after each step.
              - Execute the generated python code.
              - Synthesize information from all sources to provide a comprehensive and current report on the injury status of the player or team, with attention to the latest updates and expert opinions on recovery and potential game participation.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[CLITool.execute_cli_command],
            llm=llm
        )


# # Define the NBA General Researcher Agent
class WebSearcher(Agent):
    def __init__(self):
        super().__init__(
            role='Searches on the internet for a given query',
            goal="Browse the internet to search for a given query and provide an abstract answer to it.",
            backstory='Provies the best answer to the user query based on information provided by the tool',
            verbose=True,
            # allow_delegation=False,
            tools=[SerperDevTool()],
            llm=llm
        )


#
#
# class MarketAnalystAgent(Agent):
#     def __init__(self):
#         super().__init__(
#             role='Betting Market Analyst',
#             goal="""Analyze NBA betting market movements and trends to identify profitable betting opportunities. Only ever use verified, up to date information found from your tools.""",
#             backstory="""
#               Focused on analyzing NBA betting market movements using search tools. My method involves:
#
#                 Use DuckDuckGo to search for market trends, like 'NBA betting odds changes 2024'.
#                 Review the search summaries for patterns or significant shifts in the betting landscape.
#                 Utilize the Google search function for specific queries, such as 'impact of player X injury on NBA betting odds'. This can uncover detailed insights.
#                 Integrate data from both tools to provide a well-rounded analysis of the betting market."
#             """,
#             verbose=True,
#             allow_delegation=False,
#             tools=[search_tool, browser_tools_instance.google_search_for_answer],
#             llm=client
#         )
#
#
# class BettingAdvisorAgent(Agent):
#     def __init__(self):
#         super().__init__(
#             role='NBA Betting Advisor',
#             goal="""Provide well-reasoned betting advice combining insights from various analyses. Only ever use verified, up to date information found from your tools.""",
#             backstory="""
#                Expert in providing betting advice by synthesizing search results. My approach is:
#
#                 Search broad betting topics using DuckDuckGo, like 'NBA betting strategies 2024'.
#                 Analyze summaries to understand general advice and trends.
#                 For specific scenarios or complex questions, turn to the custom Google search. It often yields more targeted summaries.
#                 Blend insights from both searches to formulate sound, well-informed betting advice."
#             """,
#             verbose=True,
#             allow_delegation=False,
#             tools=[search_tool, browser_tools_instance.google_search_for_answer],
#             llm=client
#         )
#
#
# class ExpertOpinionAnalystAgent(Agent):
#     def __init__(self):
#         super().__init__(
#             role='Expert Opinion Analyst',
#             goal="""Provide insights based on expert NBA opinions and predictions. Use your tools to gather fresh information.""",
#             backstory="""
#                 Skilled in gathering expert NBA opinions. To tackle challenging questions, I:
#
#                 Begin with DuckDuckGo for a general search on expert predictions, like 'NBA expert predictions 2024'.
#                 Review the result summaries for key opinions and consensus.
#                 Use the Google search tool for more specific queries or to gain different perspectives.
#                 Combine the information from both searches to present a comprehensive view of expert opinions."
#             """,
#             verbose=True,
#             allow_delegation=False,
#             tools=[search_tool, browser_tools_instance.google_search_for_answer],
#             llm=client
#         )
#
#
# class NBATeamPerformanceAnalystAgent(Agent):
#     def __init__(self):
#         super().__init__(
#             role='NBA Team Performance Analyst',
#             goal="""Provide detailed analysis on the recent performance of specific NBA teams. Only ever use verified, up to date information found from your tools.""",
#             backstory="""
#                 Focused on delivering detailed performance analysis of NBA teams. My steps are:
#
#                 Use DuckDuckGo for initial searches on team performance, like 'Los Angeles Lakers performance 2024'.
#                 Scan the summaries for recent performance data and trends.
#                 For more detailed insights, especially on specific players or matches, use the Google search.
#                 Integrate the data from both searches to provide a thorough analysis of team performance."
#             """,
#             verbose=True,
#             allow_delegation=False,
#             tools=[search_tool, browser_tools_instance.google_search_for_answer],
#             llm=client
#         )
#

# Define routes for each research area
windows_route = Route(name="window_tasks", utterances=[
    "move the file report.pdf from documents folder to Desktop",
    "rename the aditya_report.pdf to report.pdf",
    "print the pdf file ask_hr",
    "Clear cache memory of the system.",
    "Go to the desktop"
    "search for the ",
    "open chrome",
    "open the whatsapp",
    "write a mail to aditya.kulshrestha@gmail.com",
])

mathematic_route = Route(name="mathematics_tasks", utterances=[
    "give me the root of 2",
    "what is 439 multiplied by 22",
    "give me the percentage increase in moving from 34 to 65",
    "what is the value of e raised to 4",
    "give me the sum of fourty four plus thirty six plus fifty seven"
])

website_route = Route(name="website_search", utterances=[
    "what is the weather today in bangalore",
    "give me top five news for today",
    "Who was Mona Lisa",
    "what are the most popular tourist attraction in India",
    "Who won the world cup 2024",
    "who won the nobel prize for 2024?",
    "give me the lyrics of mai mastana"
])

# general_route = Route(name="general", utterances=[
#     "season highlights",
#     "game recaps",
#     "upcoming NBA matchups",
#     "notable NBA news",
#     "recent trends in NBA"
# ])
#
# market_route = Route(name="market", utterances=[
#     "betting line movements",
#     "NBA betting odds",
#     "betting trends in NBA"
# ])

# advisor_route = Route(name="advisor", utterances=[
#     "betting advice for NBA games",
#     "NBA betting strategy",
#     "sports betting tips for NBA"
# ])
#
# expert_route = Route(name="expert", utterances=[
#     "expert NBA predictions",
#     "NBA game analysis",
#     "NBA match forecasts"
# ])

# team_performance_route = Route(
#     name="team_performance",
#     utterances=[
#         "team recent performance in NBA",
#         "team winning streaks in NBA",
#         "detailed team game analysis",
#         "offensive and defensive ratings of [team name]",
#         "Player performance statistics of [team name] in recent games"
#     ]
# )

# Create tasks for your agents
task1 = Task(
    description="""Break down the user request into smaller steps and most easiest flows on keyboard based actions ONLY.
    required to execute the workflow of the smaller step for the given query : {user_command}""",
    expected_output="A list of small user friendly steps to execute on windows machine. Don't return any code, "
                    "only give the steps.",
    agent=Coder()
)

code_gen = Task(
    description="Based on the provided smaller tasks, write a pyautogui code as a script",
    expected_output="Return the python script as a single output",
    agent=Coder()
)

maths_task = Task(
    description="Given a mathematical problem, break it down into simpler steps and solve it.: {user_command}",
    expected_output="Solution of the mathematical problem",
    agent=MathematicsExpert()
)

webtask = Task(
    description="Given a user query, search the internet and gather the information and answer the user: {user_command}",
    expected_output="Response to the original query",
    agent=WebSearcher()
)

# websearch = Task(
#     description="""Break down the user request into smaller steps and most easiest flows and identify which keyboards keys are
#     required to execute the workflow of the smaller step for the given query : {user_command}""",
#     expected_output="A list of small user friendly steps to execute on windows machine. Don't return any code, "
#                     "only give the steps.",
#     agent=Coder()
# )

# Initialize route layer
encoder = FastEmbedEncoder()
route_layer = RouteLayer(encoder=encoder,
                         routes=[mathematic_route, windows_route, website_route])

agent_route_map = {
    "mathematics_tasks": {
        "agent_class": MathematicsExpert,
        'tasks': [maths_task],
        "task_description": "Solve the following mathematical problem by breaking it down into simpler tasks and solve them."
    },
    "window_tasks": {
        "agent_class": Coder,
        'tasks': [task1, code_gen],
        "task_description": """
            Skilled in breaking down Windows tasks into simple, keyboard-based steps and generating corresponding PyAutoGUI code. Process:
            - Break down the task into easy, manageable steps.
            - Identify the keyboard keys needed for each step.
            - Generate the PyAutoGUI code for the identified keys.
            - Ensure the code executes smoothly.
            """
    },
    'website_search': {
        'agent_class': WebSearcher,
        'tasks': [webtask],
        "task_description": "Search for a given query on internet using the given tools"
    }
}
import datetime


@tool("Natural Language Research Tool")
def natural_language_research(request):
    """
    Provies the best assistant to the user query based on the intent they are targeting.

    Args:
        request (str): A natural language query for the request from user

    Returns:
        str: The results of the solved output as compiled by the agent.
    """

    # Determine which route to use
    route_choice = route_layer(request).name
    print(f"Route choice: {route_choice}")

    # Lookup the agent and task for the chosen route
    agent_info = agent_route_map[route_choice]
    agent_instance = agent_info["agent_class"]()
    print(agent_instance)

    # Ensure the task description includes the directive to use tools for fresh information
    # task_description = f"{agent_info['task_description']}: {request}. Remember to use your available tools to solve the given query."
    # Create the task and form the crew
    # task = Task(description=task_description, agent=agent_instance)
    task = agent_info['tasks']
    print("This is task")
    print(task)
    crew = Crew(agents=[agent_instance], tasks=task)

    # Execute the task and return the result
    result = crew.kickoff(inputs={'user_command': request})
    return result


# Example usage
if __name__ == "__main__":
    request = input("Enter your query")
    result = natural_language_research(request)
    print(result)
