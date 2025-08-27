import os
import json
import base64
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Suppress warnings and disable ChromaDB telemetry
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from chromadb.config import Settings

# Disable verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_operations.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class AgentConfig:
    """Configuration pour l'agent d'analyse d'images"""
    image_dir: str = "images"
    db_dir: str = "chroma_db_images"
    conversation_dir: str = "conversations"
    log_dir: str = "logs"
    vision_model: str = "gpt-4o"
    chat_model: str = "gpt-35-turbo"
    embedding_model: str = "text-embedding-ada-002"
    vision_temperature: float = 0.5
    chat_temperature: float = 0.7
    max_tokens: int = 1024
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_iterations: int = 5

class ImageAnalyzerAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._setup_directories()
        self._initialize_models()
        self._initialize_vector_db()
        self._initialize_state()
        self._setup_agent()

    def _setup_directories(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        Path(self.config.image_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.db_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.conversation_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_models(self) -> None:
        """Initialise tous les modèles nécessaires"""
        common_args = {
            "azure_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        }

        self.vision_model = AzureChatOpenAI(
            **common_args,
            model=self.config.vision_model,
            temperature=self.config.vision_temperature,
            max_tokens=self.config.max_tokens
        )

        self.chat_model = AzureChatOpenAI(
            **common_args,
            model=self.config.chat_model,
            temperature=self.config.chat_temperature
        )

        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model=self.config.embedding_model
        )

    def _initialize_vector_db(self) -> None:
        """Initialise la base de données vectorielle avec gestion des erreurs"""
        try:
            self.vector_db = Chroma(
                persist_directory=self.config.db_dir,
                embedding_function=self.embeddings,
                collection_name="image_analysis",
                client_settings=Settings(anonymized_telemetry=False)
            )
            logger.info("Base de données vectorielle initialisée avec succès")
        except Exception as e:
            logger.error(f"Échec de l'initialisation de ChromaDB: {e}")
            raise RuntimeError("Échec de l'initialisation de la base de données vectorielle") from e

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

    def _initialize_state(self) -> None:
        """Initialise les variables d'état de l'agent"""
        self.current_image: Optional[str] = None
        self.current_conversation_id: Optional[str] = None
        self.chat_history: List[Dict[str, Any]] = []

    def _setup_agent(self) -> None:
        """Configure l'agent avec ses outils et son exécuteur"""
        self.tools = [
            self.analyze_image_tool,
            self.list_images_tool,
            self.search_memory_tool,
            self.save_conversation_tool
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.agent = create_openai_functions_agent(
            llm=self.chat_model,
            prompt=prompt,
            tools=self.tools
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True
        )

    def _get_system_prompt(self) -> str:
        """Génère le message système en français avec un ton convivial"""
        return """Bonjour ! Je suis votre assistant IA spécialisé dans l'analyse d'images. 
        Je suis ici pour vous aider à comprendre et interpréter vos images de manière conviviale.
        
        Voici ce que je peux faire pour vous :
        - Analyser une image par son nom
        - Lister les images disponibles
        - Rechercher dans les analyses précédentes
        - Sauvegarder notre conversation
        
        Je peux me souvenir de l'image actuelle dont nous parlons.
        Je vérifie toujours si l'image existe avant de tenter de l'analyser.
        N'hésitez pas à me poser des questions supplémentaires sur les images !
        
        Je vais répondre de manière naturelle et amicale, en français uniquement.
        Après chaque analyse, je peux vous suggérer des questions pertinentes.
        """

    @property
    def analyze_image_tool(self):
        @tool
        def analyze_image(image_name: str, question: Optional[str] = None) -> str:
            """Analyse une image du répertoire images par son nom."""
            try:
                image_path = Path(self.config.image_dir) / image_name
                
                if not image_path.exists():
                    available = "\n".join(os.listdir(self.config.image_dir))
                    return f"Oups ! Je ne trouve pas l'image '{image_name}'. Voici les images disponibles :\n{available}"
                
                self.current_image = image_name
                base64_image = self._encode_image(image_path)
                
                prompt = question or "Décris-moi cette image en détail s'il te plaît"
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
                
                response = self.vision_model.invoke([HumanMessage(content=content)])
                
                # Ajout de suggestions de questions complémentaires
                analysis = response.content
                if not question:
                    follow_up = (
                        "\n\nJe peux vous aider davantage avec cette image ! "
                        "Voici des questions que vous pourriez me poser :\n"
                        "- Peux-tu identifier les éléments clés de ce document ?\n"
                        "- Y a-t-il des informations sensibles à protéger ?\n"
                        "- Pourrais-tu reformuler ce texte de manière plus simple ?"
                    )
                    analysis += follow_up
                
                self._add_to_memory(
                    content=analysis,
                    metadata={
                        "image_name": image_name,
                        "type": "image_analysis",
                        "timestamp": datetime.now().isoformat(),
                        "question": question or "Description générale"
                    }
                )
                
                return analysis
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de l'image: {e}")
                return f"Oups ! J'ai eu un problème en analysant l'image : {str(e)}"
        
        return analyze_image

    @property
    def list_images_tool(self):
        @tool
        def list_images(query: Optional[str] = None) -> str:
            """Liste toutes les images disponibles dans le répertoire images."""
            try:
                images = os.listdir(self.config.image_dir)
                if not images:
                    return "Aucune image trouvée dans le répertoire."
                return "Images disponibles :\n" + "\n".join(images)
            except Exception as e:
                logger.error(f"Erreur lors du listage des images: {e}")
                return f"Oups ! Je n'ai pas pu lister les images : {str(e)}"
        
        return list_images

    @property
    def search_memory_tool(self):
        @tool
        def search_memory(query: str, k: int = 3) -> str:
            """Recherche dans les analyses d'images et conversations précédentes."""
            try:
                docs = self.vector_db.similarity_search(query, k=k)
                if not docs:
                    return "Aucune analyse correspondante trouvée."
                
                return "\n\n".join([
                    f"Image : {doc.metadata.get('image_name', 'N/A')}\n"
                    f"Question : {doc.metadata.get('question', 'Description générale')}\n"
                    f"Analyse : {doc.page_content}\n"
                    f"Date : {doc.metadata.get('timestamp', 'inconnue')}"
                    for doc in docs
                ])
            except Exception as e:
                logger.error(f"Erreur lors de la recherche en mémoire: {e}")
                return f"Oups ! Problème lors de la recherche : {str(e)}"
        
        return search_memory

    @property
    def save_conversation_tool(self):
        @tool
        def save_conversation(notes: Optional[str] = None) -> str:
            """Sauvegarde la conversation actuelle avec des notes optionnelles."""
            try:
                if not self.chat_history:
                    return "Aucune conversation à sauvegarder."
                
                conversation_id = self._get_conversation_id()
                filename = Path(self.config.conversation_dir) / f"{conversation_id}.json"
                
                data = {
                    "id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "notes": notes,
                    "current_image": self.current_image,
                    "history": [
                        {
                            "type": type(msg).__name__,
                            "content": msg.content,
                            "timestamp": datetime.now().isoformat()
                        } 
                        for msg in self.chat_history
                    ]
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                self._add_to_memory(
                    content=f"Résumé de conversation : {notes or 'Pas de notes'}",
                    metadata={
                        "type": "conversation_summary",
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                return f"Conversation sauvegardée avec l'ID : {conversation_id}"
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde: {e}")
                return f"Oups ! Je n'ai pas pu sauvegarder la conversation : {str(e)}"
        
        return save_conversation

    def _get_conversation_id(self) -> str:
        """Génère ou retourne l'ID de conversation existant"""
        if not self.current_conversation_id:
            self.current_conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.current_conversation_id

    def _add_to_memory(self, content: str, metadata: Dict) -> None:
        """Ajoute du contenu à la mémoire vectorielle"""
        try:
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )
            self.vector_db.add_documents(docs)
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout en mémoire: {e}")

    def _encode_image(self, image_path: Path) -> str:
        """Encode une image en base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_message(self, user_input: str) -> str:
        """Traite l'entrée utilisateur et retourne une réponse claire"""
        try:
            if not self.current_conversation_id:
                self.current_conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            context_str = ""
            try:
                context = self.vector_db.similarity_search(user_input, k=2)
                context_str = "\n".join([
                    f"Analyse précédente de {doc.metadata.get('image_name', 'une image')}:\n{doc.page_content}"
                    for doc in context
                ])
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du contexte: {e}")
            
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": self.chat_history,
                "context": context_str
            })
            
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response["output"]))
            
            return response["output"]
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
            return "Oups ! Un problème est survenu lors du traitement de votre demande."

class CommandLineInterface:
    """Gère l'interface en ligne de commande pour l'agent"""
    
    @staticmethod
    def print_help() -> None:
        """Affiche les informations d'aide en français"""
        help_text = """
Commandes disponibles :
- 'aide' : Affiche ce message d'aide
- 'liste' : Montre les images disponibles
- 'analyse <nom_image>' : Analyse une image
- 'recherche <terme>' : Cherche dans les analyses précédentes
- 'sauvegarder' : Sauvegarde la conversation actuelle
- Toute question sur l'image actuelle
- 'quitter' : Quitte le programme (sauvegarde automatiquement)

Exemples :
- 'chat.jpg' (analysera l'image chat.jpg)
- 'recherche personnes' (cherche des analyses mentionnant des personnes)
- 'Quelles couleurs dominent dans coucher_soleil.jpg ?'
"""
        print(help_text.strip())

    @classmethod
    def run(cls) -> None:
        """Exécute l'interface en ligne de commande avec sortie en français"""
        analyzer = ImageAnalyzerAgent()
        print("\nBonjour ! Je suis votre assistant d'analyse d'images. Tapez 'aide' pour les instructions.")
        
        while True:
            try:
                user_input = input("\nVous : ").strip()
                
                if user_input.lower() in ['exit', 'quitter', 'bye']:
                    print("\n" + analyzer.save_conversation_tool.func("Conversation terminée par l'utilisateur"))
                    print("À bientôt ! N'hésitez pas à revenir si vous avez d'autres images à analyser.")
                    break
                elif user_input.lower() in ['help', 'aide']:
                    cls.print_help()
                    continue
                elif user_input.lower() in ['list', 'liste']:
                    print("\n" + analyzer.list_images_tool.func(""))
                    continue
                elif user_input.lower().startswith(('search ', 'recherche ')):
                    query = user_input.split(' ', 1)[1] if ' ' in user_input else ""
                    print("\n" + analyzer.search_memory_tool.func(query))
                    continue
                elif user_input.lower() in ['save', 'sauvegarder']:
                    notes = input("Notes pour cette sauvegarde (optionnel) : ").strip()
                    print("\n" + analyzer.save_conversation_tool.func(notes))
                    continue
                
                response = analyzer.process_message(user_input)
                print("\nAssistant :", response)
                
            except KeyboardInterrupt:
                print("\nÀ bientôt !")
                break
            except Exception as e:
                print("\nOups ! Un problème est survenu. Veuillez réessayer.")
                continue

if __name__ == '__main__':
    CommandLineInterface.run()