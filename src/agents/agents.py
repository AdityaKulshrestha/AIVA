import os
from textwrap import dedent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from crewai.process import Process
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv


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


class Router():
    def __init__(self):
        self.task = None