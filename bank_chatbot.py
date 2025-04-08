from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
import re
import logging
import uuid
import time
import markdown  # Import the markdown library

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
    SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Validate configuration
if not Config.GROQ_API_KEY:
    logger.critical("❌ GROQ_API_KEY is missing. Set it in .env file.")
    raise EnvironmentError("Missing GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

# Initialize Groq LLM with error handling
try:
    llm = ChatGroq(
        model=Config.MODEL_NAME,
        groq_api_key=Config.GROQ_API_KEY,
        temperature=0.3
    )
    logger.info("✅ Groq LLM initialized successfully")
except Exception as e:
    logger.critical(f"❌ LLM initialization failed: {str(e)}")
    raise

# Enhanced financial prompt template
FINANCIAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are FinBot, a certified financial assistant. Follow these rules:
1. Provide accurate, up-to-date financial information
2. Use markdown formatting for structured responses
3. Explain complex terms in simple language
4. Always maintain professional tone
5. Reject non-financial queries politely"""),
    ("human", "{user_input}")
])

# Session management with expiration
class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def get_memory(self, session_id):
        self.cleanup_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'memory': ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True
                ),
                'last_accessed': time.time()
            }
        else:
            self.sessions[session_id]['last_accessed'] = time.time()
        return self.sessions[session_id]['memory']
    
    def cleanup_sessions(self):
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session['last_accessed'] > Config.SESSION_TIMEOUT
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]

session_manager = SessionManager()

# Configure chat chain
chat_chain = RunnableWithMessageHistory(
    FINANCIAL_PROMPT | llm,
    input_messages_key="user_input",
    get_session_history=lambda session_id: session_manager.get_memory(session_id).chat_memory,
    history_messages_key="chat_history"
)

# Enhanced content filtering with regex
RESTRICTED_PATTERNS = [
    r'\b(sex|porn|violence)\b',
    r'\b(hack(ing|er)?|scam|phishing)\b',
    r'\b(terrorism|abuse)\b'
]

def contains_markdown_list(text):
    # Check if the text contains markdown list syntax
    return bool(re.search(r'(\*|\d+\.)\s', text))

@app.route("/chat", methods=["POST"])
def handle_chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Message field required"}), 400
            
        user_input = data["message"].strip()
        session_id = data.get("session_id", str(uuid.uuid4()))
        
        if not user_input:
            return jsonify({"error": "Empty message received"}), 400
            
        # Content filtering check
        for pattern in RESTRICTED_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"Blocked restricted content in session {session_id}")
                return jsonify({"response": "I specialize in financial topics only."})
        
        # Process with LLM
        try:
            memory = session_manager.get_memory(session_id)
            response = chat_chain.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            response_text = response.content
            
            # Check if the response contains markdown lists
            is_markdown_list = contains_markdown_list(response_text)
            
            # If it contains markdown lists, convert to HTML
            if is_markdown_list:
                response_text = markdown.markdown(response_text)
            
            return jsonify({
                "response": response_text,
                "session_id": session_id,
                "is_markdown_list": is_markdown_list  # Pass a flag indicating markdown list
            })
        except Exception as e:
            logger.error(f"LLM processing error: {str(e)}")
            return jsonify({"response": "Error processing your request. Please try again."}), 500
            
    except Exception as e:
        logger.exception(f"Chat endpoint error: {str(e)}")
        return jsonify({"response": "Internal server error"}), 500

@app.route("/welcome", methods=["GET"])
def welcome():
    return jsonify({
        "response": "Welcome! I'm your virtual banking assistant. Ask me anything about loans, interest rates, UPI, credit scores, investments, or savings.",
        "session_id": str(uuid.uuid4())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
