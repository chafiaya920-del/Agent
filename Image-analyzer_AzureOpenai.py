from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Dict
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
import base64

class ImageAnalyzerAgent:
    def __init__(self, image_dir="images"):
        # Set up image directory
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        # Initialize the vision model (GPT-4 with vision) using Azure
        self.vision_model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1024
        )
        
        # Initialize the chat model for text conversations using Azure
        self.chat_model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model="gpt-35-turbo",
            temperature=0.7
        )
        
        # Current image being discussed
        self.current_image = None
        
        # Set up the agent
        self.tools = [self.analyze_image_tool, self.list_images_tool]
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        
        # Chat history
        self.chat_history = []
    
    @property
    def analyze_image_tool(self):
        @tool
        def analyze_image(image_name: str, question: str = None) -> str:
            """Analyze an image from the images directory by its name. If question is provided, answer specifically about that question."""
            image_path = os.path.join(self.image_dir, image_name)
            
            if not os.path.exists(image_path):
                available = "\n".join(os.listdir(self.image_dir))
                return f"Error: Image '{image_name}' not found. Available images:\n{available}"
            
            # Set as current image
            self.current_image = image_name
            
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            
            # Prepare the prompt
            if question:
                content = [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            else:
                content = [
                    {"type": "text", "text": "What's in this image? Describe it in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            
            # Get analysis from the vision model
            response = self.vision_model.invoke([HumanMessage(content=content)])
            return response.content
        
        return analyze_image
    
    @property
    def list_images_tool(self):
        @tool
        def list_images(query: str = None) -> str:
            """List all available images in the images directory."""
            try:
                images = os.listdir(self.image_dir)
                if not images:
                    return "No images found in the images directory."
                return "Available images:\n" + "\n".join(images)
            except Exception as e:
                return f"Error listing images: {str(e)}"
        
        return list_images
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_agent(self):
        """Create the agent with tools and prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful AI assistant specialized in analyzing images. 
             Images should be placed in the '{self.image_dir}/' directory.
             When the user mentions an image name, you can analyze it in detail.
             You can remember the current image being discussed.
             Always check if the image exists before attempting to analyze it.
             You can also list available images when asked."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(
            llm=self.chat_model,
            prompt=prompt,
            tools=self.tools
        )
    
    def process_message(self, user_input: str) -> str:
        """Process user input and return response"""
        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response["output"]))
        
        return response["output"]

def print_help():
    print("\nAvailable commands:")
    print("- 'help': Show this help message")
    print("- 'list': Show available images")
    print("- 'analyze <image_name>': Analyze an image")
    print("- 'describe <image_name>': Describe an image")
    print("- Any question about the current image")
    print("- 'exit': Quit the program")
    print("\nExamples:")
    print("- 'cat.jpg' (will analyze cat.jpg)")
    print("- 'What colors are in sunset.jpg?'")
    print("- 'How many people are in this image?' (after selecting an image)")

def main():
    analyzer = ImageAnalyzerAgent()
    print("Image Analyzer Agent initialized. Type 'help' for instructions.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'help':
            print_help()
            continue
        elif user_input.lower() == 'list':
            print(analyzer.list_images_tool.func(""))
            continue
        
        response = analyzer.process_message(user_input)
        print("\nAssistant:", response)

if __name__ == '__main__':
    main()
