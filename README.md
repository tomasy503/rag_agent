# Agentic RAG using PydanticAI Framework and Crawl4AI

A powerful AI-powered assistant that combines web crawling capabilities with advanced RAG (Retrieval-Augmented Generation) to provide accurate and context-aware responses to your questions from an entire content of a domain.

## Features

- **Web Crawling**: Automatically crawls website documentation through sitemaps
- **Smart Document Processing**: 
  - Chunks documents intelligently while preserving code blocks and paragraphs
  - Generates embeddings for efficient retrieval
  - Extracts titles and summaries using GPT-4
- **Interactive Chat Interface**:
  - Streamlit-based UI with separate tabs for crawling and chat
  - Real-time streaming responses
  - Conversation history maintenance
- **RAG Implementation**:
  - Uses semantic search to find relevant documentation
  - Maintains context across conversation
  - Provides accurate, documentation-based responses

## Technologies Used

- **Frontend**: Streamlit
- **Database**: Supabase
- **AI/ML**:
  - OpenAI GPT-4 for summarization and chat
  - Alternative for another models i.e. DeepSeek for embeddings or local LLMs with Ollama
- **Web Crawling**:
  - Async Web Crawler - [Crawl4AI](https://docs.crawl4ai.com/)
  - XML Sitemap parsing
- **Other Tools**:
  - Agentic Framework - [PydanticAI](https://ai.pydantic.dev/) 
  - Python async/await for concurrent operations
  - Pydantic for data validation
  - Environment variables for configuration

## Usage

1. **Setup Environment**:
   - Ensure you have Python installed on your system.
   - Use the `pyproject.toml` file to set up dependencies with Poetry:
     ```bash
     poetry install
     poetry shell #this will activate the virtual environment
     ```

2. Set up environment variables:
   - Rename `.env.example` to `.env`
   - Edit `.env` with your API keys and preferences:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   LLM_MODEL=gpt-4o-mini  # or your preferred OpenAI model
   ```

3. **Running the Application**:
   - Make sure the website you want to crawl is the home site and has a `sitemap.xml` file available.
   - This RAG System extracts all websites from a domain.
   - Start the application using:
     ```bash
     streamlit run app.py
     ```

3. **Web Crawling**:
   - Enter the domain URL and source name in the "Web Crawler" tab.
   - Click "Start Crawling" to begin the process.

4. **Chat Interface**:
   - Use the "Chat" tab to interact with the AI assistant.
   - Ask questions about the documentation, and receive context-aware responses.

## Project Structure

- `app.py`: Main application entry point
- `src/`
  - `agent.py`: Expert agent implementation
  - `crawler.py`: Web crawler and document processor
  - `site_pages.sql`: SQL file for creating the main table to store the knowledge from the domain

## Requirements

Dependencies are managed through the `pyproject.toml` file using Poetry. To install them, run:
```bash
poetry install
```