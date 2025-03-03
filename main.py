import asyncio
import os
import re
from urllib.parse import urljoin
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig()
    )
)

async def run_search():
    agent = Agent(
        task='Search for recent research papers on sepsis detection from reputable sources like PubMed, Google Scholar, or arXiv. Provide titles and links to the most relevant papers.',
        llm=llm,
        max_actions_per_step=4,
        browser=browser,
    )

    await agent.run(max_steps=25)

    agent_output = agent.get_last_action()
    if not agent_output or "done" not in agent_output:
        raise ValueError("Agent did not complete the task successfully.")

    result_text = agent_output["done"]["text"]

    papers = []
    for line in result_text.split("\n"):
        if line.startswith("*   **"):
            match = re.match(r"\*   \*\*(.*?)\*\*: \[(.*?)\]\((.*?)\)", line)
            if match:
                title = match.group(1).strip()
                link = match.group(3).strip()
                papers.append({"title": title, "link": link})

    print("Titles and full links to the research papers:")
    for paper in papers:
        print(f"{paper['title']}: {paper['link']}")

if __name__ == '__main__':
    asyncio.run(run_search())