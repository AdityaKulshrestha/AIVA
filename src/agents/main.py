import os
from textwrap import dedent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import CodeInterpreterTool


class CLITool:
    @tool("Executor")
    def execute_cli_command(code_string: str):
        """Create and Execute code using Open Interpreter."""
        exec(code_string)
        # return result


load_dotenv()

PORTKEY_API_KEY = os.getenv('PORTKEY_API_KEY')
VIRTUAL_KEY = os.getenv('PORTKEY_VIRTUAL_KEY')
PORTKEY_API_KEY_OPENAI = os.getenv('PORTKEY_API_KEY_OPENAI')
VIRTUAL_KEY_OPENAI = os.getenv('PORTKEY_VIRTUAL_KEY_OPENAI')

llm = ChatOpenAI(
    api_key='dummy',
    base_url=PORTKEY_GATEWAY_URL,
    # model='claude-3-5-sonnet-20240620',
    default_headers=createHeaders(
        provider="anthropic",
        api_key=PORTKEY_API_KEY,
        virtual_key=VIRTUAL_KEY
    )
)
# # Working Good
# llm = ChatOpenAI(
#     api_key='dummy',
#     base_url=PORTKEY_GATEWAY_URL,
#     default_headers=createHeaders(
#         provider="openai",
#         api_key=PORTKEY_API_KEY_OPENAI,
#         virtual_key=VIRTUAL_KEY_OPENAI
#     )
# )

# print(llm.invoke("Hello world", max_tokens=50))

user_command = input("What do you want to execute: ")

# Manager Process
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
    llm=llm
)

# Planner
coder = Agent(
    role='Python Programmer',
    goal='Provide crisp and best Keyboard only PyAutoGUI code for the given commands taken from user and execute it. ',
    backstory='You are a great coder who know PyAutoGUI and can convert simple tasks into the respective commands and execute it. ',
    # allow_code_execution=True,
    tools=[CLITool.execute_cli_command],
    llm=llm,

)


task_planner = Agent(
    role='task planner',
    goal='''Break Down a task into smaller executable commands''',
    backstory='You are a task planner who plans the task for a given windows level execution.',
    llm=llm
)

# Create tasks for your agents
task1 = Task(
    description=dedent(f"Convert the following Windows task into the simplest step-by-step process, breaking it down into the easiest actions possible. Prioritize efficiency and user-friendliness, using shortcuts or automation where applicable. Take a deep breadth and think step by step : {user_command}"),
    expected_output="A list of small user friendly steps to execute on windows machine. Don't return any code, only give the steps.",
    agent=task_planner
)

task2 = Task(
    description="Based on the provided tasks, write a pyautogui code as a script",
    expected_output="Return the python script as a single output",
    agent=coder
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[task_planner, coder],
    # manager_agent=manager,
    tasks=[task1, task2],
    verbose=1,
)

result = crew.kickoff()
print("######################")
print(result)
