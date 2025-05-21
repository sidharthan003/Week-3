from autogen_agentchat.agents import AssistantAgent

class ResearcherAgent(AssistantAgent):
    """Fetches raw page HTML/text from a URL."""
    def __init__(self, name: str, browser, model_client, **kwargs):
        super().__init__(name=name, model_client=model_client, **kwargs)
        self.browser = browser

    async def _a_generate_reply(self, messages, sender, **kwargs):
        url = messages[-1]["content"].strip()
        return await self.browser.fetch(url)

class SummarizerAgent(AssistantAgent):
    """Condenses large text blobs via the TextSummarizer tool."""
    def __init__(self, name: str, summarizer, model_client, **kwargs):
        super().__init__(name=name, model_client=model_client, **kwargs)
        self.summarizer = summarizer

    async def _a_generate_reply(self, messages, sender, **kwargs):
        text = messages[-1]["content"]
        return await self.summarizer.summarize(text)
