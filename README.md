# Pydantic AI Agentic RAG

A powerful AI-powered documentation assistant that combines web crawling capabilities with advanced RAG (Retrieval-Augmented Generation) to provide accurate and context-aware responses to your questions.

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
  - Agentic Framword - [PydanticAI](https://ai.pydantic.dev/) 
  - Python async/await for concurrent operations
  - Pydantic for data validation
  - Environment variables for configuration

## Usage

1. **Setup Environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/tomasy503/rag_agent.git
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables in .env file
   OPENAI_API_KEY=your_key
   SUPABASE_URL=your_url
   SUPABASE_KEY=your_key
   DEEPSEEK_API_KEY=your_key
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Using the Application**:
   - **Crawl Documentation**:
     1. Go to the "Web Crawler" tab
     2. Enter the sitemap URL of the documentation
     3. Provide a source name (e.g., 'pydantic_docs')
     4. Click "Start Crawling"
   
   - **Chat with the Assistant**:
     1. Switch to the "Chat" tab
     2. Ask questions about the crawled documentation
     3. Receive context-aware responses based on the documentation

## Project Structure

- `app.py`: Main application entry point
- `src/`
  - `agent.py`: Expert agent implementation
  - `crawler.py`: Web crawler and document processor
  - `models/`: Model implementations

## Requirements

See `requirements.txt` for a complete list of dependencies.