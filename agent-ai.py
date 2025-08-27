import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig

# Load environment variables
load_dotenv()

# Configuration
class Config:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PERSIST_DIRECTORY = "chroma_db_web_scraping"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

# Initialize components
def initialize_components():
    # Initialize LLM (Language Model)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Initialize search tool
    search = TavilySearchResults(max_results=2)
    
    # Initialize text splitter (for chunking documents)
    text_splitter = CharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    # Initialize embeddings (for converting text to vector representations)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  
    return llm, search, text_splitter, embeddings

# Web scraping function
def scrape_and_store(urls, text_splitter, embeddings):
    try:
        # Load web content from the provided URLs
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        if not documents:
            return None, "No content was scraped from the provided URLs."
        
        # Split documents into chunks
        docs = text_splitter.split_documents(documents)
        
        # Create and store embeddings in Chroma vector store
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=Config.PERSIST_DIRECTORY
        )
        
        return db, f"Successfully scraped and stored content from {len(urls)} URLs."
    
    except Exception as e:
        return None, f"Error during scraping: {str(e)}"

# RAG prompt template (for retrieval-augmented generation)
rag_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant with access to web content and search capabilities. 
Use the following context to answer the question. If you don't know the answer, 
use your search tool to find up-to-date information.

Context: {context}

Question: {question}

Answer:
""")

# Create RAG chain
def create_rag_chain(db, llm):
    retriever = db.as_retriever()
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Main agent setup
def setup_agent():
    # Initialize components
    llm, search, text_splitter, embeddings = initialize_components()
    
    # Create tools (search and custom scraping)
    tools = [search]
    
    # Create agent with memory (so it can "remember" context)
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    
    return agent_executor, text_splitter, embeddings

# Example usage
if __name__ == "__main__":
    # Setup agent
    agent_executor, text_splitter, embeddings = setup_agent()
    
    # Create a valid configuration for streaming using RunnableConfig
    config = RunnableConfig(configurable={"thread_id": "web_scraping_thread"})  # Correct RunnableConfig format
    
    # Example conversation
    print("Agent: Hello! I'm a web scraping assistant. How can I help you today?")
    
    while True:
        # User input
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() in ['exit', 'quit']:
            print("Agent: Goodbye!")
            break
            
        # Check if the user wants to scrape a website
        if "scrape" in user_input.lower() or "extract" in user_input.lower():
            print("Agent: Please provide the URL you want to scrape:")
            url_input = input("You: ")
            urls = [url_input]  # Take the URL input from the user
            
            # Perform scraping
            db, result_msg = scrape_and_store(urls, text_splitter, embeddings)
            
            if db:
                # Create RAG chain with the new content
                rag_chain = create_rag_chain(db, agent_executor)
                print(f"Agent: {result_msg} You can now ask questions about this content.")
            else:
                print(f"Agent: {result_msg}")
            continue
        
        # Stream agent response with valid configuration
        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,  # Pass the correct RunnableConfig object here
            stream_mode="values",
        ):
            if step["messages"]:
                last_message = step["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"Agent: {last_message.content}")