import os
import asyncio
import argparse
from tools import AsyncSeleniumBrowser, TextSummarizer
from agents import ResearcherAgent, SummarizerAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat

async def run_mode(mode, urls, browser, summarizer, model_client):
    # Instantiate agents with the shared model_client
    researcher       = ResearcherAgent("researcher", browser, model_client)
    summarizer_agent = SummarizerAgent("summarizer", summarizer, model_client)

    # Pick your orchestration
    GroupClass = SelectorGroupChat if mode == "selector" else RoundRobinGroupChat
    group = GroupClass([researcher, summarizer_agent])

    for url in urls:
        print(f"\n🔍 [{mode.capitalize()}] {url}")
        # Pass the URL as a named parameter!
        summary = await group.run(task=url)
        print(f"📝 {summary}\n{'─'*60}")

async def main():
    parser = argparse.ArgumentParser(
        description="Web Research Assistant: fetch & summarize URLs"
    )
    parser.add_argument(
        "--mode",
        choices=["roundrobin", "selector"],
        default="roundrobin",
        help="Orchestration mode"
    )
    parser.add_argument("urls", nargs="+", help="URLs to process")
    args = parser.parse_args()

    # Prepare tools
    browser    = AsyncSeleniumBrowser(headless=True)
    summarizer = TextSummarizer()

    # Build a Gemini-backed OpenAI-compat client
    KEY = os.getenv("GEMINI_API_KEY")
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-pro",
        api_key=KEY
    )

    await run_mode(args.mode, args.urls, browser, summarizer, model_client)

if __name__ == "__main__":
    asyncio.run(main())
