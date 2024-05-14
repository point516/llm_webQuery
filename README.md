## Web Query System with Llama3-8b LLM via Groq API ([Try](https://t.me/siriwb_bot))

**Web Query System** based on Llama3-8b llm through Groq API, accessible through a [Telegram chatbot](https://t.me/siriwb_bot). The idea is to harness the power of the Llama3-8b language model to deliver precise and useful information directly to you with 0 hallucinations, mainly because information is scraped from the web.

### How It Works
The process involves several steps to ensure that you get the most relevant information:
1. **Optimization:** The query is optimized using the Llama3-8b model for effective Google searches.
2. **Search Execution:** A request is made to the Google Search API to retrieve data.
3. **Data Extraction:** The top three search results are scraped and the content is saved in a `.txt` file.
4. **Data Processing:** The content from the `.txt` file is embedded with the OpenAI Embeddings API, vectorized, and queried against the Llama3-8b to extract appropriate information and answer the query.

### Deployment and Technology
- **Hosting:** The system is deployed on a Google VM instance using a Docker image.
- **API Integration:** The Llama3-8b model is integrated through the Groq API. For local tests, the Ollama variant of Llama3 is used on my computer.
