import os
import asyncio
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import google.generativeai as genai

# Load & configure Gemini API key
load_dotenv()
KEY = os.getenv("GEMINI_API_KEY")
if not KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file")
genai.configure(api_key=KEY)

class AsyncSeleniumBrowser:
    """Async wrapper around Selenium ChromeDriver."""
    def __init__(self, headless: bool = True):
        opts = Options()
        opts.headless = headless
        self.driver = webdriver.Chrome(options=opts)

    async def fetch(self, url: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load, url)

    def _load(self, url: str) -> str:
        self.driver.get(url)
        return self.driver.page_source

class TextSummarizer:
    """Async Gemini-based summarizer."""
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name

    async def summarize(self, text: str, max_words: int = 200) -> str:
        prompt = (
            f"Please provide a concise summary (max {max_words} words) of the following content:\n\n{text}"
        )
        resp = await genai.chat_complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.last
