import os
import subprocess
import torch

def get_available_gpus():
    """
    Get the GPU with the most available memory.
    Returns:
        int: The GPU index with the most available memory.
    """
    try:
        # Run nvidia-smi to get GPU memory usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi error: {result.stderr}")

        # Parse the output to get free memory for each GPU
        free_memory = [int(x) for x in result.stdout.strip().split("\n")]
        best_gpu = free_memory.index(max(free_memory))  # Get the GPU with the most free memory
        return best_gpu
    except Exception as e:
        print(f"Error selecting GPU: {e}")
        return None

# Select the GPU with the most available memory
best_gpu = get_available_gpus()
if best_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Using GPU {best_gpu} with the most available memory.")
else:
    print("No GPU selected. Using default settings.")

# Check if CUDA is available and print the selected GPU
if torch.cuda.is_available():
    print(f"Selected GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Running on CPU.")

# --- Start of Imports ---
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import sqlite3
import datetime
import torch
import re
import random
import os
import json
from typing import List, Dict, Tuple, Any, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from huggingface_hub import HfApi
import asyncio
import nest_asyncio # Import nest_asyncio

# Apply the patch
nest_asyncio.apply()
# --- End of Imports ---

# --- Constants and EmotionChatbot Class Definition (Keep as before) ---
PROMPT_CONFIG = {
    "max_history_turns": 7,
    "min_history_turns": 2,
    "empathy_phrases": [
        "I hear you", "That sounds", "I can imagine",
        "It makes sense", "You must feel", "I understand",
        "That's really tough", "I appreciate you sharing"
    ],
    "prohibited_actions": [
        "suggest professional help",
        "give medical advice", 
        "assume unspecified details",
        "use clinical terms",
        "recommend third-party services",
        "provide suicide hotlines",
        "repeat what the user is asking"
    ]
}

EMOTION_TEMPLATES = {
    "sadness": [
        "This seems really heavy to carry...",
        "That pain must feel overwhelming at times...",
        "It's okay to feel down sometimes..."
    ],
    "anger": [
        "Frustration can be so consuming...",
        "It's understandable to feel that tension...",
        "That situation would test anyone's patience..."
    ],
    "anxiety": [
        "Uncertainty can be so unsettling...",
        "That worry must feel ever-present...",
        "Living with that tension sounds challenging..."
    ],
    "fear": [
        "It's natural to feel vulnerable in this situation...",
        "That uncertainty can feel threatening...",
        "Feeling exposed like that is difficult..."
    ],
    "joy": [
        "That positive energy comes through clearly...",
        "It's wonderful to hear that excitement in your words...",
        "Those moments of lightness are precious..."
    ],
    "surprise": [
        "That unexpected turn must be quite impactful...",
        "Sometimes life catches us off guard...",
        "It can be jarring when things shift so suddenly..."
    ],
    "disgust": [
        "That feeling of revulsion is completely valid...",
        "It makes sense that you'd be put off by this...",
        "That visceral reaction tells you something important..."
    ],
    "neutral": [
        "I appreciate you sharing your thoughts...",
        "Thank you for explaining your perspective...",
        "I'm here to listen to whatever you'd like to share..."
    ]
}

QUERY_TYPES = {
    "FACTUAL": "factual",
    "EMOTIONAL": "emotional",
    "COMMAND": "command",
    "GREETING": "greeting",
    "CHITCHAT": "chitchat"
}

class EmotionChatbot:
    def __init__(self, db_path: str = "chatbot_history.db"):
        # Initialize models, embeddings, db, knowledge_base, state, cache
        print("Loading models...")
        # Using lower precision can speed up loading and inference, requires bitsandbytes
        # Consider adding load_in_8bit=True or load_in_4bit=True if bitsandbytes is installed
        use_quantization = True 

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if torch.cuda.is_available():
            print("CUDA available, loading model on GPU...")
            try:
                if use_quantization:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                    print("Model loaded with 4-bit quantization.")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    print("Model loaded in float16 on GPU.")
            except Exception as e:
                print(f"Error loading model on GPU: {e}. Falling back to CPU.")
                self.model = AutoModelForCausalLM.from_pretrained(model_id)
                print("Model loaded on CPU.")
        else:
            print("CUDA not available, loading model on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            print("Model loaded on CPU.")

        # Ensure classifiers also use the best available device
        device_num = 0 if torch.cuda.is_available() else -1
        print(f"Using device {device_num} for classifiers.")
        self.query_classifier = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            return_all_scores=True,
            device=-1
        )

        # self.query_classifier = pipeline(
        # "text-classification",
        # model="cross-encoder/nli-distilroberta-base", # Smaller alternative
        # return_all_scores=True,
        # device=-1
        # )

        self.emotion_classifier = pipeline(
            "text-classification",
            model="st125338/t2e-classifier-v4",
            return_all_scores=True,
            function_to_apply="sigmoid",
            device=-1
        )

        # Embeddings model - specify device
        # embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using device '{embed_device}' for embeddings.")
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     model_kwargs={"device": embed_device}
        # )

        # Smaller embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={"device": "cpu"}
        )

        self.db_path = db_path
        self._init_db() # Renamed from initialize_database for consistency
        self.knowledge_base = self._init_knowledge_base() # Renamed from _initialize_knowledge_base

        self.conversation_state = {
            "current_intensity": 0.5,
            "trust_level": 0.1,
            "formality": 0.3,
            "last_msg_length": 0,
            "emotional_variety": 1,
            "recurring_emotion": "neutral",
            "session_turns": 0,
            "user_profile": {} # Per-user profiles could be loaded on demand
        }
        self.current_history = [] # This seems session-specific, maybe move inside chat?
        self.cache = {} # Simple cache for profiles, etc.
        print("Chatbot initialized.")


    def _init_db(self):
        # (Database initialization code - same as before)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, timestamp TEXT,
            user_message TEXT, query_type TEXT, emotions TEXT, bot_response TEXT,
            intensity REAL, context_length INTEGER, trust_level REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY, name TEXT, preferences TEXT,
            last_seen TEXT, interaction_count INTEGER, emotion_history TEXT)''')
        # Add columns if they don't exist (less verbose)
        try: cursor.execute("ALTER TABLE conversations ADD COLUMN query_type TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE user_profiles ADD COLUMN emotion_history TEXT")
        except sqlite3.OperationalError: pass
        conn.commit()
        conn.close()

    def _init_knowledge_base(self):
        # (Knowledge base initialization - same as before)
        try:
            # Consider loading documents from a file for easier management
            documents = [
                "The chatbot's name is Emotion-Aware Chatbot.",
                "The chatbot provides emotional support.",
                "The chatbot can detect user emotions.",
                "The chatbot should give direct answers to factual questions.",
                "The chatbot should be empathetic to emotional statements.",
                "The chatbot should be helpful and supportive."
            ]
            # Check if FAISS index exists, load if possible, otherwise create
            index_path = "faiss_knowledge_base.index"
            if os.path.exists(index_path):
                 print("Loading existing FAISS index...")
                 vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True) # Be cautious with allow_dangerous_deserialization
                 print("FAISS index loaded.")
            else:
                 print("Creating new FAISS index...")
                 vector_store = FAISS.from_texts(
                     documents,
                     self.embeddings,
                     metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
                 )
                 vector_store.save_local(index_path)
                 print(f"FAISS index created and saved to {index_path}.")
            return vector_store
        except Exception as e:
            print(f"Error initializing/loading knowledge base: {e}")
            return None

    # --- Other methods (_update_user_profile_db, _retrieve_knowledge, etc. remain largely the same) ---
    # Small improvements can be added, e.g., caching user profiles

    def update_knowledge_base(self, user_id: str, key: str, value: str):
        # (Same as before)
        if not self.knowledge_base: return
        try:
            document = f"User {user_id} {key}: {value}"
            self.knowledge_base.add_texts([document], metadatas=[{"source": f"user_{user_id}_{key}"}])
            # Update in-memory profile cache
            profile = self.conversation_state["user_profile"].get(user_id, {})
            profile[key] = value
            self.conversation_state["user_profile"][user_id] = profile
            # Update DB
            self._update_user_profile_db(user_id, key, value) # Ensure this method exists and is correct
        except Exception as e:
            print(f"Error updating knowledge base: {e}")

    def _update_user_profile_db(self, user_id: str, key: str, value: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT preferences, emotion_history FROM user_profiles WHERE user_id = ?", 
            (user_id,)
        )
        profile = cursor.fetchone()
        now_iso = datetime.datetime.now().isoformat()
        
        if profile:
            prefs_str, emo_hist_str = profile
            
            # Safely parse existing preferences
            try:
                preferences = json.loads(prefs_str) if prefs_str else {}
            except json.JSONDecodeError:
                preferences = {}
            
            # Update preferences with new key-value
            preferences[key] = value
    
            cursor.execute(
                "UPDATE user_profiles SET preferences = ?, last_seen = ?, interaction_count = interaction_count + 1 WHERE user_id = ?",
                (json.dumps(preferences), now_iso, user_id)
            )
        else:
            # User does not exist, create new profile entry
            preferences = {key: value}
            name = value if key == "name" else ""
            emotion_history = {}
            
            cursor.execute(
                "INSERT INTO user_profiles (user_id, name, preferences, last_seen, interaction_count, emotion_history) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, json.dumps(preferences), now_iso, 1, json.dumps(emotion_history))
            )
        
        conn.commit()
        conn.close()

    def _retrieve_knowledge(self, query: str, user_id: str = None, threshold: float = 0.3):
         # (Same as before)
         if not self.knowledge_base: return []
         try:
             # Example: Add user profile info to query context if helpful
             # query_context = f"{query}"
             # profile = self.get_user_profile(user_id) # Use cached profile if available
             # if profile.get("name"): query_context += f" User name: {profile['name']}"
             # ... add more profile context if needed ...

             docs_with_scores = self.knowledge_base.similarity_search_with_score(
                 query, k=3 # Reduced k for speed
             )
             relevant_docs = [doc.page_content for doc, score in docs_with_scores if score <= threshold] # FAISS score is distance
             # print(f"Retrieved docs: {relevant_docs}") # Debugging
             return relevant_docs or ["I'll try my best to answer based on what I know."]
         except Exception as e:
             print(f"Error retrieving knowledge: {e}")
             return ["An issue occurred while retrieving information."]


    def classify_query_type(self, query: str) -> str:
        # (Same pattern matching + classifier logic as before)
        # Can potentially optimize pattern matching slightly
        query_lower = query.lower()
        greeting_patterns = ["hello", "hi ", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(pattern in query_lower for pattern in greeting_patterns): return QUERY_TYPES["GREETING"]

        question_starters = ["what", "who", "where", "when", "why", "how", "is", "are", "was", "were", "will", "can", "could", "do ", "does ", "did "]
        is_question = any(query_lower.startswith(pattern) for pattern in question_starters) or "?" in query
        if is_question:
            emotional_words = ["feel", "sad", "happy", "angry", "anxious", "afraid", "excited", "depressed", "stressed"]
            if any(word in query_lower for word in emotional_words): return QUERY_TYPES["EMOTIONAL"]
            return QUERY_TYPES["FACTUAL"]

        command_patterns = ["tell me", "show me", "help me", "find", "search", "look up", "do ", "make", "create", "generate", "give me", "explain"]
        if any(pattern in query_lower for pattern in command_patterns): return QUERY_TYPES["COMMAND"]

        # Fallback to classifier if no pattern matches strongly
        try:
            hypotheses = [
                f"This text is a factual question asking for information: {query}",
                f"This text is an emotional statement expressing feelings: {query}",
                f"This text is a command or instruction: {query}",
                f"This text is casual conversation or small talk: {query}"
            ]
            # Note: The performance of zero-shot classification can vary. Fine-tuning might be better.
            results = self.query_classifier(query, hypotheses) # Pass only query
            # The bart-large-mnli pipeline expects sequence and candidate_labels, not multiple sequences
            # Let's adjust based on typical zero-shot usage:
            candidate_labels = ["factual question", "emotional statement", "command instruction", "casual chitchat"]
            results = self.query_classifier(query, candidate_labels)
            # results format: {'sequence': '...', 'labels': [...], 'scores': [...]}
            best_label = results['labels'][0]
            if best_label == "factual question": return QUERY_TYPES["FACTUAL"]
            if best_label == "emotional statement": return QUERY_TYPES["EMOTIONAL"]
            if best_label == "command instruction": return QUERY_TYPES["COMMAND"]
            return QUERY_TYPES["CHITCHAT"]
        except Exception as e:
            print(f"Error classifying query with model: {e}")
            return QUERY_TYPES["CHITCHAT"] # Default fallback

    def detect_emotions(self, text: str, emo_thresh: float = 0.5, top_k: int = 3) -> List[str]:
        # (Same classifier logic as before, ensure threshold is appropriate)
        try:
            # Consider caching results for very recent identical messages? Unlikely needed.
            emo_scores = self.emotion_classifier(text)[0]
            filtered = [e for e in emo_scores if e['score'] >= emo_thresh]
            sorted_top = sorted(filtered, key=lambda x: x['score'], reverse=True)[:top_k]
            labels = [e['label'] for e in sorted_top]
            if not labels: return ["neutral"]
            # Optionally update emotion history here if needed (currently done in _update_emotion_history called elsewhere)
            # self._update_emotion_history(labels[0]) # This was called here before, ensure it's called appropriately
            return labels
        except Exception as e:
            print(f"Error detecting emotions: {e}")
            return ["neutral"]

    def _update_emotion_history(self, primary_emotion: str, user_id: str):
        # (Same logic as before, uses DB)
        # Consider caching this too
        profile = self.get_user_profile(user_id) # Use cached profile
        emotion_history = profile.get("emotion_history", {})

        emotion_history[primary_emotion] = emotion_history.get(primary_emotion, 0) + 1

        # Update recurring emotion in conversation state (session-specific)
        if emotion_history:
            recurring = max(emotion_history.items(), key=lambda item: item[1])[0]
            self.conversation_state["recurring_emotion"] = recurring

        # Update profile cache and DB
        profile["emotion_history"] = emotion_history
        self.conversation_state["user_profile"][user_id] = profile
        self._save_profile_emotion_history(user_id, emotion_history)

    def _save_profile_emotion_history(self, user_id: str, emotion_history: Dict):
        # Dedicated method to save only emotion history to DB for clarity
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE user_profiles SET emotion_history = ? WHERE user_id = ?",
                (str(emotion_history), user_id)
            )
            if cursor.rowcount == 0: # User might not exist yet if first interaction
                 # If user doesn't exist, create a basic entry or handle appropriately
                 # For now, we assume profile exists from previous steps or first insert
                 print(f"Warning: User profile for {user_id} not found when updating emotion history.")
            conn.commit()
        except Exception as e:
            print(f"Error saving emotion history to DB for {user_id}: {e}")
            conn.rollback()
        finally:
            conn.close()


    def extract_user_info(self, message: str, user_id: str):
        # (Simplified: directly calls update_knowledge_base)
        # Name extraction
        sensitive_terms = ["suicid", "kill myself", "die", "death", "harm myself"]
        if any(term in message.lower() for term in sensitive_terms):
            return
        
        name_match = re.search(r"(?i)(?:my name is|i am called|i'm called|call me)\s+(\w+)", message)
        if name_match:
            name = name_match.group(1).capitalize()
            # Additional checks to avoid inappropriate names
            if len(name) > 2 and name.lower() not in ["suicidal", "depressed", "anxious", "dead"]:
                self.update_knowledge_base(user_id, "name", name)
                print(f"Extracted name: {name}")

        # Preference extraction (simple example)
        pref_matches = re.finditer(r"(?i)(?:i like|i love|i enjoy|i prefer)\s+(.+?)(?:\.|\,|\!|\?|$)", message)
        for match in pref_matches:
            preference = match.group(1).strip().lower()
            if preference and len(preference) < 50: # Basic filtering
                 # Avoid adding too many generic preferences
                 if "preference" not in preference: # Simple check
                     self.update_knowledge_base(user_id, f"preference_{preference[:10]}", preference) # Use part of pref as key
                     print(f"Extracted preference: {preference}")


    # async def save_to_database(self, user_id: str, user_message: str, query_type: str,
    #                      emotions: List[str], bot_response: str):
    #     # (Uses DB - should be async if DB driver supports it, sqlite3 doesn't natively)
    #     # For sqlite3, run blocking DB operations in a thread
    #     await asyncio.to_thread(
    #         self._save_to_database_sync, user_id, user_message, query_type, emotions, bot_response
    #     )

    async def save_to_database(self, user_id: str, user_message: str, query_type: str,
                     emotions: List[str], bot_response: str):
        # Clean inputs before saving to prevent concatenation issues
        user_message = re.sub(r'\s+', ' ', user_message).strip()
        bot_response = re.sub(r'\s+', ' ', bot_response).strip()
        
        # Run blocking DB operations in a thread
        await asyncio.to_thread(
            self._save_to_database_sync, user_id, user_message, query_type, emotions, bot_response
        )
    
    def _save_to_database_sync(self, user_id: str, user_message: str, query_type: str,
                             emotions: List[str], bot_response: str):
        # Synchronous part of saving to DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        emotions_str = ", ".join(emotions)
        try:
            cursor.execute(
                """INSERT INTO conversations
                (user_id, timestamp, user_message, query_type, emotions, bot_response, intensity, context_length, trust_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, timestamp, user_message, query_type, emotions_str, bot_response,
                    self.conversation_state["current_intensity"],
                    self._calculate_context_window(user_message), # Ensure this is efficient
                    self.conversation_state["trust_level"]
                )
            )
            # Update interaction count separately or combine if profile update logic allows
            # The profile update logic already increments count, so this might be redundant:
            # cursor.execute(
            #     "UPDATE user_profiles SET interaction_count = interaction_count + 1, last_seen = ? WHERE user_id = ?",
            #     (timestamp, user_id)
            # )
            conn.commit()
        except Exception as e:
            print(f"Error saving conversation to DB: {e}")
            conn.rollback()
        finally:
            conn.close()


    def get_user_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        # (DB access - could be cached or made async if DB supports it)
        # For simplicity, keep sync for now, run in thread if it becomes bottleneck
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Return dict-like rows
        cursor = conn.cursor()
        cursor.execute(
            """SELECT user_message, query_type, emotions, bot_response, timestamp
            FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit)
        )
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return list(reversed(history)) # Return in chronological order


    def get_user_profile(self, user_id: str) -> Dict:
        # Use cache first
        if user_id in self.cache.get("user_profiles", {}):
            # print(f"Cache hit for profile: {user_id}") # Debugging
            return self.cache["user_profiles"][user_id]

        # If not in cache, fetch from DB
        # print(f"Cache miss for profile: {user_id}, fetching from DB.") # Debugging
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, preferences, emotion_history FROM user_profiles WHERE user_id = ?",
            (user_id,)
        )
        profile_data = cursor.fetchone()
        conn.close()

        profile = {"name": "", "preferences": {}, "emotion_history": {}}
        if profile_data:
            name, prefs_str, emo_hist_str = profile_data
            profile["name"] = name
            try: profile["preferences"] = eval(prefs_str) if prefs_str else {}
            except: profile["preferences"] = {}
            try: profile["emotion_history"] = eval(emo_hist_str) if emo_hist_str else {}
            except: profile["emotion_history"] = {}

        # Store in cache
        if "user_profiles" not in self.cache: self.cache["user_profiles"] = {}
        self.cache["user_profiles"][user_id] = profile
        return profile


    def _calculate_context_window(self, user_message: str) -> int:
        # (Heuristic calculation - seems okay)
        msg_len = len(user_message)
        if msg_len == 0: return PROMPT_CONFIG["min_history_turns"]

        features = {
            "exclamation": user_message.count("!") / max(1, msg_len / 50.0),
            "question": user_message.count("?") / max(1, msg_len / 50.0),
            "length": min(1.0, len(user_message.split()) / 100.0),
            "caps_ratio": sum(1 for c in user_message if c.isupper()) / float(msg_len)
        }
        intensity_score = (
            0.3 * features["exclamation"] + 0.2 * features["question"] +
            0.3 * features["length"] + 0.2 * features["caps_ratio"]
        )
        # Update conversation intensity (consider averaging or decaying more slowly?)
        self.conversation_state["current_intensity"] = (self.conversation_state["current_intensity"] * 0.5 + intensity_score * 0.5)

        # Dynamic window size based on intensity
        dynamic_turns = int(PROMPT_CONFIG["min_history_turns"] + intensity_score * (PROMPT_CONFIG["max_history_turns"] - PROMPT_CONFIG["min_history_turns"]))
        return min(PROMPT_CONFIG["max_history_turns"], max(PROMPT_CONFIG["min_history_turns"], dynamic_turns))

    def _format_history_context(self, history: List[Dict], include_emotions: bool = True) -> str:
        # (Formatting history - seems okay)
        if not history: return "No previous conversation."
        context = "Previous conversation:\n"
        for turn in history:
             # Ensure keys exist before accessing
             user_msg = turn.get('user_message', '[message unavailable]')
             bot_resp = turn.get('bot_response', '[response unavailable]')
             context += f"User: {user_msg}\n"
             if include_emotions and turn.get('emotions'):
                 context += f"[Emotions detected: {turn['emotions']}]\n"
             context += f"Assistant: {bot_resp}\n\n" # Add extra newline for clarity
        return context.strip() # Remove trailing newlines


    def _build_base_prompt_template(self, system_content: str) -> str:
         # (Adding prohibited actions - seems okay)
         prohibited = "\n\nIMPORTANT: DO NOT:\n" + "\n".join([f"- {action}" for action in PROMPT_CONFIG["prohibited_actions"]])
         return f"{system_content}\n{prohibited}"


    def _generate_raw_response(self, prompt: str, query_type: str) -> str:
        # (Model generation - should be run in thread if model isn't async compatible)
        # Transformers pipeline and generate are typically blocking CPU/GPU operations
        # Running them in a thread prevents blocking the main async loop
        # print(f"Generating response for query type: {query_type}") # Debugging
        # print(f"Prompt length: {len(prompt)}") # Debugging
        # print(f"Prompt sample: {prompt[:500]}...") # Debugging

        # Adjust parameters based on type
        if query_type == QUERY_TYPES["FACTUAL"]:
            temperature = 0.3
            top_p = 0.85
            max_tokens = 150 # Shorter for factual
        elif query_type == QUERY_TYPES["EMOTIONAL"]:
            temperature = 0.75 # Slightly higher for more varied emotional resp.
            top_p = 0.9
            max_tokens = 250 # Allow longer emotional resp.
        elif query_type == QUERY_TYPES["GREETING"]:
            temperature = 0.6
            top_p = 0.9
            max_tokens = 50 # Short greetings
        else: # COMMAND or CHITCHAT
            temperature = 0.6
            top_p = 0.9
            max_tokens = 200

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device) # Added truncation

            # Generate response
            # Use torch.inference_mode() for efficiency
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, # Pass tokenized inputs directly
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, # Ensure EOS token is set
                    repetition_penalty=1.15, # Slightly increased penalty
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

            # Decode response
            # Ensure decoding skips prompt tokens correctly
            input_length = inputs["input_ids"].shape[1]
            # response = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
            response = self.tokenizer.decode(output[0][input_length:], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
            print(f"Raw response: {response}")

            # Force garbage collection after generation
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            return response

        except Exception as e:
            print(f"Error generating raw response: {e}")
            # Provide a more informative fallback
            return "I encountered an issue while generating a response. Please try rephrasing your message."


    def _process_response(self, raw_response: str, query_type: str, emotions: List[str]) -> str:
        # Remove potential instruction remnants if model repeats prompt structure
        inst_end = raw_response.find("[/INST]")
        if inst_end != -1:
            response = raw_response[inst_end + len("[/INST]"):].strip()
        else:
            response = raw_response.strip()
    
        # Remove common bot prefixes
        for prefix in ["Assistant:", "Chatbot:", "AI:", "Response:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
    
        # Improved text cleaning - fix spacing issues
        # Replace multiple spaces with single space
        response = re.sub(r'\s+', ' ', response)
        
        # Fix missing spaces after punctuation
        response = re.sub(r'([.!?,;])([A-Za-z])', r'\1 \2', response)
        
        # Fix concatenated words - more difficult, use dictionary check if possible
        # Basic fix for common patterns (camelCase splitting)
        response = re.sub(r'([a-z])([A-Z])', r'\1 \2', response)
    
        # Inject emotional phrases if needed
        if query_type == QUERY_TYPES["EMOTIONAL"]:
            response = self._insert_emotional_phrases(response, emotions)
    
        # Apply safety filters
        response = self._apply_safety_filters(response)
    
        # Length constraints (adjust as needed)
        if query_type == QUERY_TYPES["FACTUAL"] and len(response.split()) > 60:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            response = ' '.join(sentences[:3]) # Limit to first 3 sentences for factual
    
        # Overall max length constraint
        max_len_chars = 800
        if len(response) > max_len_chars:
            response = response[:max_len_chars]
            # Try to end on a complete sentence
            last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_punct > 0:
                response = response[:last_punct + 1]
            else: # If no punctuation, just add ellipsis
                response += "..."
    
        # Ensure response is not empty
        return response if response else "I'm not sure how to respond to that. Could you rephrase?"


    def _insert_emotional_phrases(self, response: str, emotions: List[str]) -> str:
        # (Injecting empathy - same logic, ensure templates are diverse)
        # Check if response already contains empathetic phrases
        has_empathy = any(phrase.lower() in response.lower() for phrase in PROMPT_CONFIG["empathy_phrases"])

        # Add template if response is short or lacks empathy
        needs_phrase = len(response.split()) < 15 or not has_empathy

        if needs_phrase and emotions and emotions[0] != "neutral":
            primary_emotion = emotions[0]
            if primary_emotion in EMOTION_TEMPLATES:
                template = random.choice(EMOTION_TEMPLATES[primary_emotion])
                # Prepend template for better flow
                response = f"{template} {response}"
        return response


    def _apply_safety_filters(self, response: str) -> str:
        # Initialize filtered_response with the input response
        filtered_response = response
        
        # Use word boundaries for more precise matching
        prohibited_patterns = [
            # Expand these patterns to be more comprehensive
            r"\bseek\s+(?:professional|medical|psychiatric)\s+help\b",
            r"\b(?:therapist|counselor|psychologist|psychiatrist)\b", 
            r"\bdoctor\b", r"\bmedical\s+advice\b", r"\bdiagnosis\b",
            r"\bhotline\b", r"\bemergency\s+services\b", r"\bcall\s+\d{3}\b",
            r"\bsuicide\s+(?:hotline|prevention|ideation|line|crisis)\b", 
            r"\bcrisis\s+(?:line|center|resource|service|intervention)\b",
            r"\bmental\s+health\s+(?:professional|service|center|help)\b",
            r"\bself-harm\b", r"\bkilling\s+myself\b",
            r"\b\d{1,3}[-\.]?\d{3}[-\.]?\d{4}\b",  # Phone numbers
            r"\bwww\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # Websites
            # Add crisis terms
            r"\bnational\s+(?:suicide|crisis|emergency)\b",
            r"\blifeline\b", r"\bsamaritans\b", r"\bbefrienders\b"
        ]
                
        # Complete removal instead of replacement for crisis services
        for pattern in prohibited_patterns:
            filtered_response = re.sub(pattern, "", filtered_response, flags=re.IGNORECASE)
        
        # Replace any remaining mentions or suggestions with safer alternatives
        filtered_response = re.sub(r"(?:call|text|contact|visit|reach out to).*?(?:support|help|resource|service)", 
                           "consider talking to someone you trust", filtered_response, flags=re.IGNORECASE)
        
        # For suicidal content specifically, provide a safe replacement message
        if re.search(r"\b(?:suicid|kill\s+(?:myself|yourself)|harm\s+(?:myself|yourself))\b", filtered_response, re.IGNORECASE):
            return "I understand you're going through a difficult time. I'm here to listen and chat with you. Would you like to talk about something else that's on your mind?"
        
        return filtered_response
        # # Replacement text - should guide towards general support if possible
        # replacement = "getting support"
        # filtered_response = response
        # for pattern in prohibited_patterns:
        #     filtered_response = re.sub(pattern, replacement, filtered_response, flags=re.IGNORECASE)

        # # Generic check for overly clinical language (example)
        # clinical_terms = [r"\bclinically\b", r"\bdisorder\b", r"\bsymptom\b"]
        # for term in clinical_terms:
        #      filtered_response = re.sub(term, "feeling", filtered_response, flags=re.IGNORECASE)

        # # Prevent suggesting specific external services/apps unless explicitly allowed
        # if "recommend" in filtered_response.lower() and ("app" in filtered_response.lower() or "service" in filtered_response.lower()):
        #     # Basic filter - might need refinement
        #     filtered_response = re.sub(r"recommend.*?(app|service)", "mention options for support", filtered_response, flags=re.IGNORECASE)

        # return filtered_response

    def _update_conversation_state(self, user_msg: str, bot_response: str):
         # (Updating state like trust, intensity - seems okay)
         # Trust builds slowly, decays slightly
         self.conversation_state["trust_level"] = min(0.95, self.conversation_state["trust_level"] * 0.98 + 0.02)
         # Intensity decays over time
         self.conversation_state["current_intensity"] *= 0.90
         # Formality might decrease slightly over longer conversations
         if self.conversation_state["session_turns"] > 5:
             self.conversation_state["formality"] = max(0.1, self.conversation_state["formality"] * 0.95)

    # --- System Prompt Building Methods (_build_system_prompt, etc. - Keep as before) ---
    # Ensure they use the latest conversation_state and profile info

    def _build_system_prompt(self, user_profile: Dict, query_type: str, emotions: List[str]) -> str:
        # Selects the appropriate prompt builder based on query type
        builders = {
            QUERY_TYPES["FACTUAL"]: self._build_factual_system_prompt,
            QUERY_TYPES["EMOTIONAL"]: self._build_emotional_system_prompt,
            QUERY_TYPES["COMMAND"]: self._build_command_system_prompt,
            QUERY_TYPES["GREETING"]: self._build_greeting_system_prompt,
            QUERY_TYPES["CHITCHAT"]: self._build_chitchat_system_prompt,
        }
        builder = builders.get(query_type, self._build_chitchat_system_prompt) # Default to chitchat

        # Pass necessary arguments
        if query_type == QUERY_TYPES["EMOTIONAL"]:
            return builder(user_profile, emotions)
        else:
            return builder(user_profile)

    # --- Specific Prompt Builders ---
    def _build_factual_system_prompt(self, user_profile: Dict) -> str:
        user_name = user_profile.get("name")
        name_context = f" The user's name is {user_name}." if user_name else ""
        # Add preference context if available and relevant
        prefs = user_profile.get("preferences", {})
        pref_context = ""
        if prefs:
            likes = [val for key, val in prefs.items() if key.startswith("preference")]
            if likes: pref_context = f" They like {', '.join(likes[:2])}." # Limit preferences shown
    
        system_content = f"""You are AskHelper, a precise and factual conversational assistant.{name_context}{pref_context}
Your goal is to answer questions accurately and concisely based ONLY on the provided context or your general knowledge if no context is given.

IMPORTANT SAFETY RULES:
- NEVER provide information on self-harm, suicide methods, or ways to harm oneself or others
- NEVER share crisis resources, hotlines, phone numbers, or medical services
- NEVER suggest professional help for emotional or mental health concerns
- If asked about harmful topics, respond with "I'd prefer to help with something else" and suggest a neutral topic
- NEVER provide instructions for creating weapons, dangerous substances, or illegal activities
- For questions about harmful content, reply with factual information about safer related topics instead

RESPONSE GUIDELINES:
- Answer directly
- Be brief (1-3 sentences usually)
- If you don't know, say "I don't have that information"
- Stick to the facts requested
- Maintain a helpful but neutral tone"""
        return self._build_base_prompt_template(system_content)

    def _build_emotional_system_prompt(self, user_profile: Dict, emotions: List[str]) -> str:
        user_name = user_profile.get("name")
        name_context = f" You are speaking with {user_name}." if user_name else ""
        emotion_str = ", ".join(emotions) if emotions and emotions[0] != 'neutral' else "some unexpressed feelings"
        recurring = self.conversation_state.get('recurring_emotion', 'neutral')
        recurring_context = f" Sometimes they feel {recurring}." if recurring != 'neutral' and recurring not in emotions else ""
    
        system_content = f"""You are EmpathyBot, a warm, understanding, and supportive companion.{name_context}
The user seems to be feeling {emotion_str}.{recurring_context}

CRITICAL SAFETY PROTOCOL:
- NEVER provide information about self-harm, suicide methods, or ways to cause harm
- NEVER share crisis resources, hotlines, or specific phone numbers, even with good intentions
- NEVER suggest professional help, therapists, counselors, or medical services for emotional crises
- If the user mentions harming themselves, express empathy and IMMEDIATELY change the subject to something positive
- For concerning messages, respond with warmth but redirect: "I care about you. What else is on your mind today?"
- NEVER diagnose conditions or use clinical terminology
- Focus on listening and validation, not solving their problems

YOUR APPROACH:
- Use 1-2 empathetic phrases (e.g., "I hear you," "That sounds tough")
- Validate their emotions without judgment (e.g., "It's okay to feel that way")
- Be present and attentive. Keep responses concise but warm (2-4 sentences)
- Reflect understanding without suggesting solutions
- Maintain a calm, caring, and non-clinical tone"""
        return self._build_base_prompt_template(system_content)

    def _build_command_system_prompt(self, user_profile: Dict) -> str:
        user_name = user_profile.get("name")
        name_context = f" User's name: {user_name}." if user_name else ""
    
        system_content = f"""You are TaskExec, an efficient assistant focused on completing instructions.{name_context}
Your goal is to understand the user's command and execute it accurately, EXCEPT:
- NEVER provide ANY information about self-harm, suicide methods, or ways to harm oneself
- NEVER share crisis resources, hotlines, or phone numbers
- NEVER suggest professional help or medical services for emotional crises
- If the user mentions harming themselves, respond with empathy but IMMEDIATELY change the subject
- For harmful commands, reply: "I'd be happy to help with something else instead."
- For other tasks, provide clear, step-by-step information
- Be concise and direct
- Use a helpful and professional tone"""
        return self._build_base_prompt_template(system_content)

    def _build_greeting_system_prompt(self, user_profile: Dict) -> str:
        user_name = user_profile.get("name")
        greeting = f"Hello {user_name}!" if user_name else "Hello!"
    
        system_content = f"""You are GreetBot, friendly and welcoming.{greeting}
Your goal is to respond warmly to greetings.

SAFETY GUIDELINES:
- NEVER provide information on harmful activities, self-harm, or dangerous topics
- NEVER share crisis resources, hotlines, phone numbers, or suggest professional help
- If the greeting contains concerning content, respond warmly but generically
- Redirect concerning greetings to neutral topics like "How are you today?" or "What's on your mind?"
- Keep a positive, uplifting tone in all interactions

GREETING APPROACH:
- Keep it brief and positive (1-2 sentences)
- Be personable and inviting
- Ask a simple opening question like "How can I help you today?" or "How are you feeling?"
- Use a cheerful tone"""
        return self._build_base_prompt_template(system_content)

    def _build_chitchat_system_prompt(self, user_profile: Dict) -> str:
        user_name = user_profile.get("name")
        name_context = f" You're chatting with {user_name}." if user_name else ""
        prefs = user_profile.get("preferences", {})
        pref_context = ""
        if prefs:
             likes = [val for key, val in prefs.items() if key.startswith("preference")]
             if likes: pref_context = f" They enjoy things like {', '.join(likes[:2])}."
    
        system_content = f"""You are ChitChatPal, a friendly and engaging conversationalist.{name_context}{pref_context}
Your goal is to maintain a natural, light-hearted conversation.

SAFETY PROTOCOLS:
- NEVER provide information about self-harm, suicide methods, or ways to cause harm
- NEVER share crisis resources, hotlines, phone numbers, or suggest medical/professional help
- If the conversation turns to concerning topics, gently redirect to positive subjects
- For mentions of self-harm, respond with care but quickly change the topic: "I'm here to chat. Tell me about something you enjoy?"
- NEVER use clinical terms or attempt to diagnose conditions
- Always maintain a casual, supportive tone even with difficult topics

CONVERSATION STYLE:
- Respond casually and briefly (1-3 sentences)
- Show interest, ask follow-up questions occasionally
- Be personable and maintain flow
- Use a friendly, relaxed tone
- Focus on the user's interests and positive topics"""
        return self._build_base_prompt_template(system_content)


    # --- Main Chat Method ---
    async def chat(self, user_message: str, user_id: str = "default_user") -> str:
        # Main interaction logic
        try:
            # 1. Classify Query
            # Run blocking classifier in thread
            query_type = await asyncio.to_thread(self.classify_query_type, user_message)
            print(f"Query type: {query_type}")

            # 2. Detect Emotions (only if needed)
            emotions = ["neutral"]
            if query_type in [QUERY_TYPES["EMOTIONAL"], QUERY_TYPES["CHITCHAT"]]:
                # Run blocking classifier in thread
                emotions = await asyncio.to_thread(self.detect_emotions, user_message)
                print(f"Detected emotions: {', '.join(emotions)}")
                # Update emotion history based on detected emotions
                if emotions and emotions[0] != "neutral":
                     self._update_emotion_history(emotions[0], user_id) # This updates state and queues DB save

            # 3. Extract User Info (runs quickly, no need for thread?)
            # Consider if regex is slow on very long messages
            self.extract_user_info(user_message, user_id)

            # 4. Prepare Context (History, Knowledge)
            # Get history (sync DB call for now, wrap if slow)
            history_limit = self._calculate_context_window(user_message)
            history = self.get_user_history(user_id, limit=history_limit)
            include_emotions_in_hist = query_type == QUERY_TYPES["EMOTIONAL"]
            history_context = self._format_history_context(history, include_emotions_in_hist)

            # Retrieve knowledge if factual (sync RAG call, wrap if slow)
            knowledge_results = []
            if query_type == QUERY_TYPES["FACTUAL"]:
                knowledge_results = self._retrieve_knowledge(user_message, user_id)
            knowledge_context = "Context:\n" + "\n".join(f"- {item}" for item in knowledge_results) + "\n\n" if knowledge_results else ""

            # 5. Get User Profile (uses cache)
            user_profile = self.get_user_profile(user_id)

            # 6. Build Prompt
            system_prompt = self._build_system_prompt(user_profile, query_type, emotions)
            # Combine all parts for the final prompt
            # Use the specific format expected by Mistral Instruct
            # Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format
            # Construct messages list for tokenizer chat template if available/preferred
            # Or stick to manual string formatting:
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{knowledge_context}{history_context}User: {user_message.strip()} [/INST]"


            # 7. Generate Response (run blocking model call in thread)
            raw_response = await asyncio.to_thread(self._generate_raw_response, prompt, query_type)

            # 8. Process Response (fast, no thread needed)
            processed_response = self._process_response(raw_response, query_type, emotions)

            #processed_response = await asyncio.to_thread(self._check_response_safety, user_message, processed_response)
            
            # 9. Update state (fast, no thread needed)
            self._update_conversation_state(user_message, processed_response)
            self.conversation_state["session_turns"] += 1

            # 10. Save interaction to DB (runs DB write in thread)
            await self.save_to_database(user_id, user_message, query_type, emotions, processed_response)

            # Append to current in-memory history (if needed for immediate context)
            self.current_history.append(HumanMessage(content=user_message))
            self.current_history.append(AIMessage(content=processed_response))
            # Limit in-memory history size if it grows too large
            max_mem_hist = 20 # Example limit
            if len(self.current_history) > max_mem_hist * 2 :
                 self.current_history = self.current_history[-(max_mem_hist*2):]

            return processed_response

        except Exception as e:
            # Log the full error for debugging
            import traceback
            print(f"Error in chat method: {e}\n{traceback.format_exc()}")
            return "I encountered an unexpected issue. Please try again."

# --- Main Execution Block ---
async def main_loop():
    # Initialize chatbot outside the loop
    chatbot = EmotionChatbot()
    print("Chatbot: Hello! How can I help you today?")
    user_id = "test_user_async" # Example user ID

    while True:
        try:
            # Run blocking input() in a separate thread
            user_input = await asyncio.to_thread(input, "You: ")

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye! Take care.")
                break

            # Call the async chat method
            response = await chatbot.chat(user_input, user_id)
            print(f"Chatbot: {response}")

        except (KeyboardInterrupt, EOFError): # Handle Ctrl+C or EOF
            print("\nChatbot: Goodbye! Take care.")
            break
        except Exception as e:
            print(f"\nAn error occurred during interaction: {e}")
            # Optionally add a small delay or attempt recovery
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        print("Starting chatbot...")
        # Use asyncio.run to start the main async loop
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nExiting chatbot.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred at the top level: {e}\n{traceback.format_exc()}")