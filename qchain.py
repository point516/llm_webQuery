from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import google_search, extract_text_from_p_tags
from dotenv import load_dotenv
import os

#Variables
def query_chain(prompt):
    load_dotenv()
    api_key = os.getenv("SEARCH_API")
    cse_id = os.getenv("SEARCH_ID")
    llm = ChatOllama(model='llama3', temperature=0)
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    # 2 PromptTemplates for LLama3-8b, one for optimising the query for google search, other for querying the parsed text and give an answer
    opt_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a professional google searcher. 
                    Listen to the user's question and construct an appropriate and efficient google search from this prompt. Give only google search as 
                    an answer, nothing else.
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    Here is the prompt: {question}
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                """,
        input_variables=["question"],
    )

    ans_prompt = PromptTemplate(
        template = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a professional query system that gives the most accurate answer to the provided prompt. Analyze the prompt, understand the question
                and give an accurate answer based on the provided context. Some of the information in the context might be completely irrelevant, disregard
                it and take into account only information relevant to the question. Be precise and concise enough to answer the question correctly.
                <context>
                {context}
                </context>
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here is the question:
                {input}
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """,
        input_variables=["context","input"],
    )

    #llm chain for optimizing the query
    siri = opt_prompt | llm | StrOutputParser()
    query = siri.invoke({"question": prompt})
    print(query)  

    #API call to Google Search, getting 3 top results
    try:
        results = google_search(query[1:-1], api_key, cse_id, num=3)
    except Exception as e:
        print("An error occurred:", e)

    #Creating the file or emptying it
    open('parsed.txt','w').close() # to empty the file

    #Parse top-3 websites and save to .txt file
    for result in results:
        text = extract_text_from_p_tags(result['link'], user_agent)
        with open('parsed.txt', 'a', encoding='utf-8') as file:
            file.write(text)

    #Read parsed text, split it and make a query system with Llama3
    with open('parsed.txt', 'r', encoding='utf-8') as file:
        data = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    texts = text_splitter.create_documents([data])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    document_chain = create_stuff_documents_chain(llm, ans_prompt) # this provides variable {context}
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input":prompt})
    return response['answer']