from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Literal, TypedDict

import logfire
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
from supabase import Client

from agent import ExpertAgent, ExpertAgentDeps
from crawler import WebCrawler

# Load environment variables
load_dotenv()


# Initialize clients
# client = OpenAIModel(
#     api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
# )

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase: Client = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="if-token-present")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


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


async def run_agent_with_streaming(user_input: str, agent: Agent):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = ExpertAgentDeps(
        supabase=supabase,
        openai_client=openai_client,
    )

    # Run the agent in a stream
    async with agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[
            :-1
        ],  # pass entire conversation so far
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


async def crawl_website(sitemap_url: str, source_name: str, crawler: WebCrawler):
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

            total_urls = len(urls)

            def update_progress_callback():
                nonlocal current
                current += 1
                update_progress(current, total_urls, progress_text)

            current = 0
            await crawler.crawl_parallel(
                urls, progress_callback=update_progress_callback
            )
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
    if "source_name" not in st.session_state:
        st.session_state.source_name = None

    expert_agent = ExpertAgent(st.session_state.source_name)
    agent = expert_agent.agent
    crawler = WebCrawler(st.session_state.source_name)

    # Create tabs for crawler and chat
    tab1, tab2 = st.tabs(["Web Crawler", "Chat"])

    # Crawler Tab
    with tab1:
        st.header("Web Crawler")
        sitemap_url = st.text_input("Enter sitemap URL:")
        source_name = st.text_input("Enter source name (e.g., 'pydantic_ai_docs'):")

        if st.button("Start Crawling") and sitemap_url and source_name:
            st.session_state.source_name = source_name
            success = await crawl_website(sitemap_url, source_name, crawler)
            if success:
                st.session_state.messages = []  # Reset chat history for new source

    # Chat Tab
    with tab2:
        st.header("Chat Interface")

        if not st.session_state.source_name:
            st.warning("Please crawl a documentation site first in the Crawler tab.")
            return

        # Initialize chat history in session state if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display all messages from the conversation so far
        # Each message is either a ModelRequest or ModelResponse.
        # We iterate over their parts to decide how to display them.
        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                for part in msg.parts:
                    display_message_part(part)

        # Chat input for the user
        user_input = st.chat_input(
            "What questions do you have about the documentation?"
        )

        if user_input:
            # We append a new request to the conversation explicitly
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )

            # Display user prompt in the UI
            with st.chat_message("user"):
                st.markdown(user_input)

            # Display the assistant's partial response while streaming
            with st.chat_message("assistant"):
                # Actually run the agent now, streaming the text
                await run_agent_with_streaming(user_input, agent)


if __name__ == "__main__":
    asyncio.run(main())
