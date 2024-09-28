import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI  # Updated import for OpenAI Chat models
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain  # Updated import
from utils import google_search, extract_text_from_p_tags  # Assuming these utilities are correctly defined

# Load environment variables
load_dotenv()

# Variables
def query_chain(prompt):
    # OpenAI API key should be set in the .env file as OPENAI_API_KEY
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

    # Initialize OpenAI Chat Model
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # Use the desired OpenAI model
        temperature=0,  # Set temperature to 0 for deterministic outputs
        openai_api_key=openai_api_key
    )

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


    # Define Prompt Templates
    opt_prompt = PromptTemplate(
        template="""You are a professional Google searcher.
Listen to the user's question and construct an appropriate and efficient Google search query from this prompt. 
Provide only the Google search query as the answer, nothing else.

User's question: {question}""",
        input_variables=["question"],
    )

    ans_prompt = PromptTemplate(
        template="""
You are a professional query system that provides the most accurate answer to the provided prompt. 
Analyze the prompt, understand the question, and give an accurate answer based on the provided context. 
Some of the information in the context might be completely irrelevant; disregard it and take into account only information relevant to the question. 
Be precise and concise enough to answer the question correctly.

Context:
{context}

Question:
{input}
""",
        input_variables=["context", "input"],
    )

    # Create a chain for optimizing the query
    siri = opt_prompt | llm | StrOutputParser()
    query = siri.invoke({"question": prompt}).strip()
    print(f"Optimized Query: {query}")

    # API call to Google Search, getting top 3 results
    try:
        results = google_search(query, os.getenv("SEARCH_API"), os.getenv("SEARCH_ID"), num=3)
    except Exception as e:
        print("An error occurred during Google Search:", e)
        return 'Your question gives no results in Google Searcg API. Try another question, please'

    # Clear the 'parsed.txt' file
    with open('parsed.txt', 'w', encoding='utf-8') as file:
        pass  # Simply opening in write mode clears the file

    # Parse top-3 websites and save to 'parsed.txt'
    for result in results:
        text = extract_text_from_p_tags(result['link'], user_agent)
        with open('parsed.txt', 'a', encoding='utf-8') as file:
            file.write(text + "\n")  # Ensure separation between documents

    # Read parsed text and prepare for embedding
    with open('parsed.txt', 'r', encoding='utf-8') as file:
        data = file.read()

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.create_documents([data])

    # Generate embeddings and create a FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create the document chain with the answer prompt
    document_chain = create_stuff_documents_chain(llm, ans_prompt) # this provides variable {context}

    # Initialize the retriever
    retriever = vectorstore.as_retriever()

    # Create the retrieval-based QA chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Invoke the retrieval chain with the user's prompt
    response = retrieval_chain.invoke({"input": prompt})

    return response['answer']


if __name__ == "__main__":
    question = input()
    answer = query_chain(question)
    print(answer)