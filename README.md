## Web Query System with ChatGPT4o-mini with OpenaAI API ([Try](https://t.me/siriwb_bot))

**Web Query System** based on ChatGPT4o-mini, accessible through a [Telegram chatbot](https://t.me/siriwb_bot). The idea is to harness the power and cheapness of the ChatGPT4o-mini language model to deliver precise, up-to-date and useful information directly to you with minimum hallucinations, mainly because information is scraped from the web.

### How It Works
The process involves several steps to ensure that you get the most relevant information:
1. **Optimization:** The query is optimized using the aforementioned LLM model for effective Google searches.
2. **Search Execution:** A request is made to the Google Search API to retrieve data.
3. **Data Extraction:** The top three search results are scraped and the content is saved in a `.txt` file.
4. **Data Processing:** The content from the `.txt` file is embedded with the OpenAI Embeddings API, vectorized, and queried against the 4o-mini model to extract appropriate information and answer the query.

### Deployment and Technology
- **Hosting:** The system is deployed on a Google VM instance using a Docker image.
- **API Integration:** The 4o-mini model is integrated through the OpenaAI API.
