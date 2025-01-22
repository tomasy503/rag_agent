from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client, create_client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
)


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


class WebCrawler:
    def __init__(self, supabase_client: Client, source_name: str):
        self.source_name = source_name
        self.supabase = supabase_client
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        self.crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async def get_sitemap_urls(self, url: str) -> List[str]:
        """Get URLs from a sitemap."""
        sitemap_url = f"{url}/sitemap.xml"
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()

            # Parse the XML
            root = ElementTree.fromstring(response.content)

            # Extract all URLs from the sitemap
            namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

            return urls
        except Exception as e:
            print(f"Error fetching sitemap: {e}")
            return []

    def chunk_text(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into chunks, respecting code blocks and paragraphs."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + chunk_size

            # If we're at the end of the text, just take what's left
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find a code block boundary first (```)
            chunk = text[start:end]
            code_block = chunk.rfind("```")
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block

            # If no code block, try to break at a paragraph
            elif "\n\n" in chunk:
                # Find the last paragraph break
                last_break = chunk.rfind("\n\n")
                if (
                    last_break > chunk_size * 0.3
                ):  # Only break if we're past 30% of chunk_size
                    end = start + last_break

            # If no paragraph break, try to break at a sentence
            elif ". " in chunk:
                # Find the last sentence break
                last_period = chunk.rfind(". ")
                if (
                    last_period > chunk_size * 0.3
                ):  # Only break if we're past 30% of chunk_size
                    end = start + last_period + 1

            # Extract chunk and clean it up
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk
            start = max(start + 1, end)

        return chunks

    async def get_title_and_summary(self, chunk: str, url: str) -> Dict[str, str]:
        """Extract title and summary using GPT-4."""
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""

        try:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
                    },  # Send first 1000 chars for context
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {
                "title": "Error processing title",
                "summary": "Error processing summary",
            }

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI."""
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536  # Return zero vector on error

    async def process_chunk(
        self, chunk: str, chunk_number: int, url: str
    ) -> ProcessedChunk:
        """Process a single chunk of text."""
        # Get title and summary
        extracted = await self.get_title_and_summary(chunk, url)

        # Get embedding
        embedding = await self.get_embedding(chunk)

        # Create metadata
        metadata = {
            "source": self.source_name,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
        }

        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted["title"],
            summary=extracted["summary"],
            content=chunk,  # Store the original chunk content
            metadata=metadata,
            embedding=embedding,
        )

    async def insert_chunk(self, chunk: ProcessedChunk):
        """Insert a processed chunk into Supabase."""
        try:
            data = {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }

            result = supabase.table("site_pages").insert(data).execute()
            print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
            print("Insert result:", result.data)  # Log the result of the insertion
            return result
        except Exception as e:
            print(f"Error inserting chunk: {e}")
            return None

    async def clear_table(self):
        try:
            response = supabase.table("site_pages").delete().neq("id", 0).execute()
            if response.status_code == 200:
                print("Table cleared successfully.")
            else:
                print("Failed to clear table:", response.json())
        except Exception as e:
            print("Error:", e)

    async def process_and_store_document(self, url: str, markdown: str):
        """Process a document and store its chunks in parallel."""
        # Split into chunks
        chunks = self.chunk_text(markdown)

        # Process chunks in parallel
        tasks = [self.process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
        processed_chunks = await asyncio.gather(*tasks)

        # Store chunks in parallel
        insert_tasks = [self.insert_chunk(chunk) for chunk in processed_chunks]
        await asyncio.gather(*insert_tasks)

    async def crawl_parallel(
        self, urls: List[str], max_concurrent: int = 5, progress_callback=None
    ):
        """Crawl multiple URLs in parallel with a concurrency limit."""
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

        # Clear the table at the start of a new crawl session
        await self.clear_table()

        # Create the crawler instance
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()

        try:
            # Create a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_url(url: str):
                async with semaphore:
                    result = await crawler.arun(
                        url=url, config=crawl_config, session_id="session1"
                    )
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        await self.process_and_store_document(
                            url, result.markdown_v2.raw_markdown
                        )
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")

                    # Call the progress callback if provided
                    if progress_callback:
                        progress_callback()

            # Process all URLs in parallel with limited concurrency
            await asyncio.gather(*[process_url(url) for url in urls])
        finally:
            await crawler.close()
