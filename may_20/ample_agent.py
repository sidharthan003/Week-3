from autogen_agentchat.agents import AssistantAgent
from google.generativeai import GenerativeModel

client = GenerativeModel("gemini-1.5-pro")
agent = AssistantAgent(
  name="Test",
  model_client=client,
  description="desc",
  system_message="sys",
  tools=[]
)
print("OK")
