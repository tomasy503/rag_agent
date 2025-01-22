from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from typing import List

import logfire  # Import logfire
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class ExpertAgentDeps:
    supabase: Client
    openai_client: AsyncOpenAI


class ExpertAgent:
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.system_prompt = f"""
            You are an expert at researching websites - you have access to all the documentation,
            including examples, an API reference, and other resources to help answer relevant questions.

            Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

            Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

            When you first look at the documentation, always start with RAG.
            Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.
            Do not use your general knowledge to answer the user's question. Use exclusively the documentation and the tools provided.
            If you don't have the answer in the documentation, always let the user know.

            Always let the user know when you didn't find the answer in the documentation or the right URL - be honest. 
            """
        self.agent = Agent(
            model,
            system_prompt=self.system_prompt,
            deps_type=ExpertAgentDeps,
            retries=2,
            tools=[
                self.retrieve_relevant_documentation,
                self.list_documentation_pages,
                self.get_page_content,
            ],
        )

    async def get_embedding(self, text: str, openai_client: AsyncOpenAI) -> List[float]:
        """Get embedding vector from OpenAI."""
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536  # Return zero vector on error

    async def retrieve_relevant_documentation(
        self, ctx: RunContext[ExpertAgentDeps], user_query: str
    ) -> str:
        """
        Retrieve relevant documentation chunks based on the query with RAG.

        Args:
            ctx: The context including the Supabase client and OpenAI client
            user_query: The user's question or query

        Returns:
            A formatted string containing the top 5 most relevant documentation chunks
        """
        try:
            # Get the embedding for the query
            query_embedding = await self.get_embedding(
                user_query, ctx.deps.openai_client
            )

            # Query Supabase for relevant documents
            result = ctx.deps.supabase.rpc(
                "match_site_pages",
                {
                    "query_embedding": query_embedding,
                    "match_count": 5,
                    "filter": {"source": self.source_name},
                },
            ).execute()

            if not result.data:
                return f"No relevant documentation found for {self.source_name}."

            # Format the results
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"""
    # {doc['title']}

    {doc['content']}
    """
                formatted_chunks.append(chunk_text)

            # Join all chunks with a separator
            return "\n\n---\n\n".join(formatted_chunks)

        except Exception as e:
            print(f"Error retrieving documentation: {e}")
            return f"Error retrieving documentation: {str(e)}"

    async def list_documentation_pages(
        self, ctx: RunContext[ExpertAgentDeps]
    ) -> List[str]:
        """
        Retrieve a list of all available Pydantic AI documentation pages.

        Returns:
            List[str]: List of unique URLs for all documentation pages
        """
        try:
            # Query Supabase for unique URLs where source is source_name
            result = (
                ctx.deps.supabase.from_("site_pages")
                .select("url")
                .eq("metadata->>source", self.source_name)
                .execute()
            )

            if not result.data:
                return []

            # Extract unique URLs
            urls = sorted(set(doc["url"] for doc in result.data))
            return urls

        except Exception as e:
            print(f"Error retrieving documentation pages: {e}")
            return []

    async def get_page_content(self, ctx: RunContext[ExpertAgentDeps], url: str) -> str:
        """
        Retrieve the full content of a specific documentation page by combining all its chunks.

        Args:
            ctx: The context including the Supabase client
            url: The URL of the page to retrieve

        Returns:
            str: The complete page content with all chunks combined in order
        """
        try:
            # Query Supabase for all chunks of this URL, ordered by chunk_number
            result = (
                ctx.deps.supabase.from_("site_pages")
                .select("title, content, chunk_number")
                .eq("url", url)
                .eq("metadata->>source", self.source_name)
                .order("chunk_number")
                .execute()
            )

            if not result.data:
                return f"No content found for URL: {url}"

            # Format the page with its title and all chunks
            page_title = result.data[0]["title"].split(" - ")[0]  # Get the main title
            formatted_content = [f"# {page_title}\n"]

            # Add each chunk's content
            for chunk in result.data:
                formatted_content.append(chunk["content"])

            # Join everything together
            return "\n\n".join(formatted_content)

        except Exception as e:
            print(f"Error retrieving page content: {e}")
            return f"Error retrieving page content: {str(e)}"
