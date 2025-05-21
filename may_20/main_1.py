import asyncio
import os
import subprocess
from tempfile import NamedTemporaryFile

import google.generativeai as genai
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool  # for wrapping our linter :contentReference[oaicite:0]{index=0}

# 1) Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2) Adapter so AssistantAgent sees .model_info & async .generate()
class GeminiClientAdapter:
    def __init__(self, model_name: str):
        self.model = genai.GenerativeModel(model_name)
        self.model_info = {"function_calling": True}

    async def generate(self, messages):
        return await self.model.generate_content_async(messages)

# 3) Our custom Agent class
class GeminiAgent(AssistantAgent):
    def __init__(self, *, name: str, model_client, **kwargs):
        super().__init__(name=name, model_client=model_client, **kwargs)
        self.model = model_client

    async def _a_generate_reply(self, messages, sender, **kwargs):
        formatted = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = await self.model.generate(formatted)
        return resp.text

# 4) Lint function to wrap
async def lint_code(code: str) -> str:
    """Runs pylint on Python code and returns a lint report."""
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        path = tmp.name
    proc = subprocess.run(
        ["pylint", path, "--disable=all", "--enable=E,W,C,R"],
        capture_output=True,
        text=True,
    )
    return proc.stdout

async def main():
    # 5) Start local code executor
    work_dir = os.path.abspath(".")
    executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    await executor.start()
    python_executor = PythonCodeExecutionTool(executor)

    # 6) Wrap our lint function as a proper FunctionTool
    linter = FunctionTool(
        func=lint_code,
        description=lint_code.__doc__,
        name="linter",
    )

    # 7) Build a single Gemini client adapter
    gemini_client = GeminiClientAdapter("gemini-1.5-pro")

    # 8) Common tools list
    tools = [python_executor, linter]

    # 9) Instantiate the two agents with tools passed in
    coder = GeminiAgent(
        name="Coder",
        model_client=gemini_client,
        description="Writes Python code based on instructions.",
        system_message="You are a Python coder. Write clear, PEP8-compliant code.",
        tools=tools,
    )
    debugger = GeminiAgent(
        name="Debugger",
        model_client=gemini_client,
        description="Analyzes and debugs Python code using pylint.",
        system_message="You are a debugger. Provide linting feedback and suggest fixes.",
        tools=tools,
    )

    # 10) Round‐robin between them
    team = RoundRobinGroupChat(agents=[coder, debugger])
    prompt = (
        "Task: Write and debug a Python function to compute Fibonacci numbers. "
        "Coder writes the function; Debugger lints and refines it."
    )
    result = await team.chat(prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
