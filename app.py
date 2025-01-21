from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Literal, TypedDict

# import logfire
import streamlit as st

# Load environment variables
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# from pydantic_ai.models.openai import OpenAIModel
from supabase import Client, create_client

from src.agent import ExpertAgent, ExpertAgentDeps
from src.crawler import WebCrawler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
# client = OpenAIModel(
#     api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
# )
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
)

source_name = None
# Initialize WebCrawler
crawler = WebCrawler(supabase, source_name)

# Configure logfire to suppress warnings (optional)
# logfire.configure(send_to_logfire="never")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str
    
def truncate_table():
    try:
        # Directly execute SQL to truncate the table
        response = supabase.postgres.query("TRUNCATE TABLE site_pages;").execute()
        if response.status_code == 200:
            print("Table truncated successfully.")
        else:
            print("Failed to truncate table:", response.json())
    except Exception as e:
        print("Error:", e)


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)
    # tool-call
    elif part.part_kind == "tool-call":
        with st.chat_message("assistant"):
            st.markdown(f"**Tool Call**: {part.content}")
    # tool-return
    elif part.part_kind == "tool-return":
        with st.chat_message("assistant"):
            st.markdown(f"**Tool Result**: {part.content}")


def convert_message_history(messages):
    """Convert message history to the correct format."""
    converted = []
    for msg in messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            converted.append(msg)
        elif isinstance(msg, dict):
            # Convert dict to ModelRequest/ModelResponse
            if msg["role"] == "user":
                converted.append(
                    ModelRequest(parts=[UserPromptPart(content=msg["content"])])
                )
            else:
                converted.append(
                    ModelResponse(parts=[TextPart(content=msg["content"])])
                )
    return converted


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = ExpertAgentDeps(
        supabase=supabase,
        openai_client=openai_client,
        source_name=st.session_state.source_name,
    )

    # Create the agent with the current source name
    expert = ExpertAgent(st.session_state.source_name)

    # Convert message history to proper format
    message_history = convert_message_history(st.session_state.messages[:-1])

    # Run the agent in a stream
    async with expert.agent.run_stream(
        user_input,
        deps=deps,
        message_history=message_history,  # Use converted message history
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [
            msg
            for msg in result.new_messages()
            if not (
                hasattr(msg, "parts")
                and any(part.part_kind == "user-prompt" for part in msg.parts)
            )
        ]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def crawl_website(sitemap_url: str, source_name: str):
    """Crawl a website using its sitemap URL."""
    try:
        # Get URLs from sitemap
        urls = await crawler.get_sitemap_urls(sitemap_url)
        if urls:
            progress_text = "Crawling in progress..."
            progress_bar = st.progress(0, text=progress_text)

            # Create a progress callback
            def update_progress(current: int, total: int, message: str):
                progress = float(current) / float(total)
                progress_bar.progress(progress, text=f"{message} ({current}/{total})")

            # Crawl and store the content
            await crawler.crawl_parallel(urls)
            progress_bar.progress(1.0, text="Crawling completed!")
            st.success(f"Successfully crawled and stored content for {source_name}")
            return True
        else:
            st.error("No URLs found in the sitemap")
            return False
    except Exception as e:
        st.error(f"Error during crawling: {str(e)}")
        return False


async def main():
    st.title("Pydantic AI Agentic RAG")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "source_name" not in st.session_state:
        st.session_state.source_name = None

    # Create tabs for crawler and chat
    tab1, tab2 = st.tabs(["Web Crawler", "Chat"])

    # Crawler Tab
    with tab1:
        st.header("Web Crawler")
        sitemap_url = st.text_input("Enter sitemap URL:")
        source_name = st.text_input("Enter source name (e.g., 'pydantic_ai_docs'):")

        if st.button("Start Crawling") and sitemap_url and source_name:
            st.session_state.source_name = source_name
            success = await crawl_website(sitemap_url, source_name)
            if success:
                st.session_state.messages = []  # Reset chat history for new source

    # Chat Tab
    with tab2:
        st.header("Chat Interface")

        if not st.session_state.source_name:
            st.warning("Please crawl a documentation site first in the Crawler tab.")
            return

        # Display existing messages
        for msg in st.session_state.messages:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                for part in msg.parts:
                    display_message_part(part)
            elif isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                with st.chat_message(role):
                    st.markdown(content)

        # Chat input
        user_input = st.chat_input(
            f"Ask questions about {st.session_state.source_name}"
        )

        if user_input:
            # Add user message to history as ModelRequest
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Display assistant response
            with st.chat_message("assistant"):
                await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
