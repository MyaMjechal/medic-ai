import os
import torch
import re
import random
import os
import json
import sqlite3
import datetime
import asyncio
import aiosqlite
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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
        "repeat what the user is asking",
        "emojis"
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
    def __init__(self, model, tokenizer, db_path: str = "chatbot_history.db"):
        print("[Chatbot] Initializing EmotionChatbot...")
        self.model = model
        self.tokenizer = tokenizer

        device_num = 0 if torch.cuda.is_available() else -1
        print(f"Using device {device_num} for classifiers.")
        
        self.query_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_num
        )

        self.emotion_classifier = pipeline(
            "text-classification",
            model="st125338/t2e-classifier-v4",
            return_all_scores=True,
            device=device_num
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={"device": "cpu"}
        )

        self.db_path = db_path
        self._init_db()
        self.knowledge_base = self._init_knowledge_base()

        self.conversation_state = {
            "current_intensity": 0.5,
            "trust_level": 0.1,
            "formality": 0.3,
            "last_msg_length": 0,
            "emotional_variety": 1,
            "recurring_emotion": "neutral",
            "session_turns": 0,
            "user_profile": {}
        }
        self.current_history = []
        self.cache = {"user_profiles": {}}
        print("Chatbot initialized.")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, timestamp TEXT,
            user_message TEXT, query_type TEXT, emotions TEXT, bot_response TEXT,
            intensity REAL, context_length INTEGER, trust_level REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY, name TEXT, preferences TEXT,
            last_seen TEXT, interaction_count INTEGER, emotion_history TEXT)''')
        try: cursor.execute("ALTER TABLE conversations ADD COLUMN query_type TEXT")
        except sqlite3.OperationalError: pass
        try: cursor.execute("ALTER TABLE user_profiles ADD COLUMN emotion_history TEXT")
        except sqlite3.OperationalError: pass
        conn.commit()
        conn.close()

    def _init_knowledge_base(self):
        try:
            documents = [
                "The chatbot's name is Emotion-Aware Chatbot.",
                "The chatbot provides emotional support.",
                "The chatbot can detect user emotions.",
                "The chatbot should give direct answers to factual questions.",
                "The chatbot should be empathetic to emotional statements.",
                "The chatbot should be helpful and supportive."
            ]
            index_path = "faiss_knowledge_base.index"
            # if os.path.exists(index_path):
            #      print("Loading existing FAISS index...")
            #      vector_store = FAISS.load_local(index_path, self.embeddings)
            #      print("FAISS index loaded.")
            # else:
            #      print("Creating new FAISS index...")
            #      vector_store = FAISS.from_texts(
            #          documents,
            #          self.embeddings,
            #          metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
            #      )
            #      vector_store.save_local(index_path)
            #      print(f"FAISS index created and saved to {index_path}.")
            # return vector_store
            if os.path.exists(index_path):
                print("Loading existing FAISS index…")
                vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True   # <<–– add this
                )
                print("FAISS index loaded.")
            else:
                # 2) otherwise build it from scratch
                print("Creating new FAISS index…")
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

    def update_knowledge_base(self, user_id: str, key: str, value: str):
        if not self.knowledge_base: return
        try:
            document = f"User {user_id} {key}: {value}"
            self.knowledge_base.add_texts([document], metadatas=[{"source": f"user_{user_id}_{key}"}])
            profile = self.conversation_state["user_profile"].get(user_id, {})
            profile[key] = value
            self.conversation_state["user_profile"][user_id] = profile
            self._update_user_profile_db(user_id, key, value)
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
            
            try:
                preferences = json.loads(prefs_str) if prefs_str else {}
            except json.JSONDecodeError:
                preferences = {}
            
            preferences[key] = value
    
            cursor.execute(
                "UPDATE user_profiles SET preferences = ?, last_seen = ?, interaction_count = interaction_count + 1 WHERE user_id = ?",
                (json.dumps(preferences), now_iso, user_id)
            )
        else:
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
        if not self.knowledge_base: return []
        try:
            docs_with_scores = self.knowledge_base.similarity_search_with_score(
                query, k=3
            )
            relevant_docs = [doc.page_content for doc, score in docs_with_scores if score <= threshold]
            return relevant_docs or ["I'll try my best to answer based on what I know."]
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return ["An issue occurred while retrieving information."]

    def classify_query_type(self, query: str) -> str:
        query_lower = query.lower()
        greeting_patterns = ["hello", "hi ", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(pattern in query_lower for pattern in greeting_patterns): 
            return QUERY_TYPES["GREETING"]
    
        question_starters = ["what", "who", "where", "when", "why", "how", "is", "are", "was", "were", "will", "can", "could", "do ", "does ", "did "]
        is_question = any(query_lower.startswith(pattern) for pattern in question_starters) or "?" in query
        if is_question:
            emotional_words = ["feel", "sad", "happy", "angry", "anxious", "afraid", "excited", "depressed", "stressed"]
            if any(word in query_lower for word in emotional_words): 
                return QUERY_TYPES["EMOTIONAL"]
            return QUERY_TYPES["FACTUAL"]
    
        command_patterns = ["tell me", "show me", "help me", "find", "search", "look up", "do ", "make", "create", "generate", "give me", "explain"]
        if any(pattern in query_lower for pattern in command_patterns): 
            return QUERY_TYPES["COMMAND"]
    
        try:
            candidate_labels = ["factual question", "emotional statement", "command instruction", "casual chitchat"]
            results = self.query_classifier(
                query,
                candidate_labels=candidate_labels
            )
            
            best_label = results["labels"][0]
            if best_label == "factual question": 
                return QUERY_TYPES["FACTUAL"]
            if best_label == "emotional statement": 
                return QUERY_TYPES["EMOTIONAL"]
            if best_label == "command instruction": 
                return QUERY_TYPES["COMMAND"]
            return QUERY_TYPES["CHITCHAT"]
        except Exception as e:
            print(f"Error classifying query with model: {e}")
            return QUERY_TYPES["CHITCHAT"]
        
    def detect_emotions(self, text: str, emo_thresh: float = 0.5, top_k: int = 3) -> List[str]:
        try:
            emo_scores = self.emotion_classifier(text)[0]
            filtered = [e for e in emo_scores if e['score'] >= emo_thresh]
            sorted_top = sorted(filtered, key=lambda x: x['score'], reverse=True)[:top_k]
            labels = [e['label'] for e in sorted_top]
            if not labels: 
                return ["neutral"]
            return labels
        except Exception as e:
            print(f"Error detecting emotions: {e}")
            return ["neutral"]

    def _update_emotion_history(self, primary_emotion: str, user_id: str):
        profile = self.get_user_profile(user_id)
        emotion_history = profile.get("emotion_history", {})

        emotion_history[primary_emotion] = emotion_history.get(primary_emotion, 0) + 1

        if emotion_history:
            recurring = max(emotion_history.items(), key=lambda item: item[1])[0]
            self.conversation_state["recurring_emotion"] = recurring

        profile["emotion_history"] = emotion_history
        self.conversation_state["user_profile"][user_id] = profile
        self._save_profile_emotion_history(user_id, emotion_history)

    def _save_profile_emotion_history(self, user_id: str, emotion_history: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE user_profiles SET emotion_history = ? WHERE user_id = ?",
                (json.dumps(emotion_history), user_id)
            )
            if cursor.rowcount == 0:
                 print(f"Warning: User profile for {user_id} not found when updating emotion history.")
            conn.commit()
        except Exception as e:
            print(f"Error saving emotion history to DB for {user_id}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def extract_user_info(self, message: str, user_id: str):
        sensitive_terms = ["suicid", "kill myself", "die", "death", "harm myself"]
        if any(term in message.lower() for term in sensitive_terms):
            return
        
        name_match = re.search(r"(?i)(?:my name is|i am called|i'm called|call me)\s+(\w+)", message)
        if name_match:
            name = name_match.group(1).capitalize()
            if len(name) > 2 and name.lower() not in ["suicidal", "depressed", "anxious", "dead"]:
                self.update_knowledge_base(user_id, "name", name)
                print(f"Extracted name: {name}")

        pref_matches = re.finditer(r"(?i)(?:i like|i love|i enjoy|i prefer)\s+(.+?)(?:\.|\,|\!|\?|$)", message)
        for match in pref_matches:
            preference = match.group(1).strip().lower()
            if preference and len(preference) < 50:
                 if "preference" not in preference:
                     self.update_knowledge_base(user_id, f"preference_{preference[:10]}", preference)
                     print(f"Extracted preference: {preference}")

    async def save_to_database(self, user_id: str, user_message: str, query_type: str,
                     emotions: List[str], bot_response: str):
        user_message = re.sub(r'\s+', ' ', user_message).strip()
        bot_response = re.sub(r'\s+', ' ', bot_response).strip()
        
        await self._save_to_database_async(user_id, user_message, query_type, emotions, bot_response)
    
    async def _save_to_database_async(self, user_id: str, user_message: str, query_type: str,
                             emotions: List[str], bot_response: str):
        async with aiosqlite.connect(self.db_path) as db:
            timestamp = datetime.datetime.now().isoformat()
            emotions_str = ", ".join(emotions)
            try:
                await db.execute(
                    """INSERT INTO conversations
                    (user_id, timestamp, user_message, query_type, emotions, bot_response, intensity, context_length, trust_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id, timestamp, user_message, query_type, emotions_str, bot_response,
                        self.conversation_state["current_intensity"],
                        self._calculate_context_window(user_message),
                        self.conversation_state["trust_level"]
                    )
                )
                await db.commit()
            except Exception as e:
                print(f"Error saving conversation to DB: {e}")

    def get_user_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """SELECT user_message, query_type, emotions, bot_response, timestamp
            FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit)
        )
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return list(reversed(history))

    def get_user_profile(self, user_id: str) -> Dict:
        if user_id in self.cache["user_profiles"]:
            return self.cache["user_profiles"][user_id]

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
            try: 
                profile["preferences"] = json.loads(prefs_str) if prefs_str else {}
            except: 
                profile["preferences"] = {}
            try: 
                profile["emotion_history"] = json.loads(emo_hist_str) if emo_hist_str else {}
            except: 
                profile["emotion_history"] = {}

        self.cache["user_profiles"][user_id] = profile
        return profile

    def _calculate_context_window(self, user_message: str) -> int:
        msg_len = len(user_message)
        if msg_len == 0: return PROMPT_CONFIG["min_history_turns"]

        features = {
            "exclamation": user_message.count("!") / max(1, msg_len / 50.0),
            "question": user_message.count("?") / max(1, msg_len / 50.0),
            "length": min(1.0, len(user_message.split()) / 100.0),
            "caps_ratio": sum(1 for c in user_message if c.isupper()) / float(max(1, msg_len))
        }
        intensity_score = (
            0.3 * features["exclamation"] + 0.2 * features["question"] +
            0.3 * features["length"] + 0.2 * features["caps_ratio"]
        )
        self.conversation_state["current_intensity"] = (self.conversation_state["current_intensity"] * 0.5 + intensity_score * 0.5)

        dynamic_turns = int(PROMPT_CONFIG["min_history_turns"] + intensity_score * (PROMPT_CONFIG["max_history_turns"] - PROMPT_CONFIG["min_history_turns"]))
        return min(PROMPT_CONFIG["max_history_turns"], max(PROMPT_CONFIG["min_history_turns"], dynamic_turns))

    def _format_history_context(self, history: List[Dict], include_emotions: bool = True) -> str:
        if not history: return "No previous conversation."
        context = "Previous conversation:\n"
        for turn in history:
             user_msg = turn.get('user_message', '[message unavailable]')
             bot_resp = turn.get('bot_response', '[response unavailable]')
             context += f"User: {user_msg}\n"
             if include_emotions and turn.get('emotions'):
                 context += f"[Emotions detected: {turn['emotions']}]\n"
             context += f"Assistant: {bot_resp}\n\n"
        return context.strip()

    def _build_base_prompt_template(self, system_content: str) -> str:
         prohibited = "\n\nIMPORTANT: DO NOT:\n" + "\n".join([f"- {action}" for action in PROMPT_CONFIG["prohibited_actions"]])
         return f"{system_content}\n{prohibited}"

    def _generate_raw_response(self, prompt: str, query_type: str) -> str:
        if query_type == QUERY_TYPES["FACTUAL"]:
            temperature = 0.3
            top_p = 0.85
            max_tokens = 200
        elif query_type == QUERY_TYPES["EMOTIONAL"]:
            temperature = 0.75
            top_p = 0.9
            max_tokens = 300
        elif query_type == QUERY_TYPES["GREETING"]:
            temperature = 0.6
            top_p = 0.9
            max_tokens = 150
        else:
            temperature = 0.6
            top_p = 0.9
            max_tokens = 200

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_beams=4,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                output[0][input_length:], 
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True
            ).strip()

            if not response.endswith(('.', '!', '?')) and len(response) > 3:
                response += "..."
            print(f"Raw response: {response[:100]}...")

            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            return response

        except Exception as e:
            print(f"Error generating raw response: {e}")
            return "I encountered an issue while generating a response. Please try rephrasing your message."

    def _process_response(self, raw_response: str, query_type: str, emotions: List[str]) -> str:
        try:
            inst_end = raw_response.find("[/INST]")
            if inst_end != -1:
                response = raw_response[inst_end + len("[/INST]"):].strip()
            else:
                response = raw_response.strip()
        
            for prefix in ["Assistant:", "Chatbot:", "AI:", "Response:"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
        
            response = re.sub(r'\s+', ' ', response)
            response = re.sub(r'([.!?,;])([A-Za-z])', r'\1 \2', response)
            response = re.sub(r'([a-z])([A-Z])', r'\1 \2', response)
        
            if query_type == QUERY_TYPES["EMOTIONAL"]:
                response = self._insert_emotional_phrases(response, emotions)
        
            response = self._apply_safety_filters(response)
        
            if query_type == QUERY_TYPES["FACTUAL"] and len(response.split()) > 60:
                sentences = re.split(r'(?<=[.!?])\s+', response)
                response = ' '.join(sentences[:3])
        
            max_len_chars = 800
            if len(response) > max_len_chars:
                response = response[:max_len_chars]
                last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_punct > 0:
                    response = response[:last_punct + 1]
                else:
                    response += "..."
        
            return response if response else "I'm not sure how to respond to that. Could you rephrase?"
        except Exception as e:
            print(f"Error processing response: {e}")
            return "I had a small hiccup while responding. What would you like to talk about?"

    def _insert_emotional_phrases(self, response: str, emotions: List[str]) -> str:
        has_empathy = any(phrase.lower() in response.lower() for phrase in PROMPT_CONFIG["empathy_phrases"])
        needs_phrase = len(response.split()) < 15 or not has_empathy

        if needs_phrase and emotions and emotions[0] != "neutral":
            primary_emotion = emotions[0]
            if primary_emotion in EMOTION_TEMPLATES:
                template = random.choice(EMOTION_TEMPLATES[primary_emotion])
                response = f"{template} {response}"
        return response

    def _apply_safety_filters(self, response: str) -> str:
        filtered_response = response
        
        prohibited_patterns = [
            r"\bseek\s+(?:professional|medical|psychiatric)\s+help\b",
            r"\b(?:therapist|counselor|psychologist|psychiatrist)\b", 
            r"\bdoctor\b", r"\bmedical\s+advice\b", r"\bdiagnosis\b",
            r"\bhotline\b", r"\bemergency\s+services\b", r"\bcall\s+\d{3}\b",
            r"\bsuicide\s+(?:hotline|prevention|ideation|line|crisis)\b", 
            r"\bcrisis\s+(?:line|center|resource|service|intervention)\b",
            r"\bmental\s+health\s+(?:professional|service|center|help)\b",
            r"\bself-harm\b", r"\bkilling\s+myself\b",
            r"\b\d{1,3}[-\.]?\d{3}[-\.]?\d{4}\b",
            r"\bwww\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            r"\bnational\s+(?:suicide|crisis|emergency)\b",
            r"\blifeline\b", r"\bsamaritans\b", r"\bbefrienders\b"
        ]
                
        for pattern in prohibited_patterns:
            filtered_response = re.sub(pattern, "", filtered_response, flags=re.IGNORECASE)
        
        filtered_response = re.sub(r"(?:call|text|contact|visit|reach out to).*?(?:support|help|resource|service)", 
                       "consider talking to someone you trust", filtered_response, flags=re.IGNORECASE)
        
        if re.search(r"\b(?:suicid|kill\s+(?:myself|yourself)|harm\s+(?:myself|yourself))\b", filtered_response, re.IGNORECASE):
            return "I understand you're going through a difficult time. I'm here to listen and chat with you. Would you like to talk about something else that's on your mind?"
        
        return filtered_response

    def _update_conversation_state(self, user_msg: str, bot_response: str):
         self.conversation_state["trust_level"] = min(0.95, self.conversation_state["trust_level"] * 0.98 + 0.02)
         self.conversation_state["current_intensity"] *= 0.90
         if self.conversation_state["session_turns"] > 5:
             self.conversation_state["formality"] = max(0.1, self.conversation_state["formality"] * 0.95)

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
    
        system_content = f"""You are MedicAI, a precise and conversational assistant.{name_context}{pref_context}
Your goal is to answer questions accurately and concisely based ONLY on the provided context or your general knowledge if no context is given.

IMPORTANT SAFETY RULES:
- NEVER provide information on self-harm, suicide methods, or ways to harm oneself or others
- NEVER share crisis resources, hotlines, phone numbers, or medical services
- NEVER suggest professional help for emotional or mental health concerns
- If asked about harmful topics, respond with "I'd prefer to help with something else" and suggest a neutral topic
- NEVER provide instructions for creating weapons, dangerous substances, or illegal activities
- For questions about harmful content, reply factual information with safer related topics instead

RESPONSE GUIDELINES:
- Answer directly
- Be brief (1-2 sentences usually)
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
    
        system_content = f"""You are MedicAI, a warm, understanding, and supportive companion.{name_context}
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
- Use 1 empathetic phrases (e.g., "I hear you," "That sounds tough")
- Validate their emotions without judgment (e.g., "It's okay to feel that way")
- Be present and attentive. Keep responses concise but warm (2-4 sentences)
- Reflect understanding without suggesting solutions
- Maintain a calm, caring, and non-clinical tone"""
        return self._build_base_prompt_template(system_content)

    def _build_command_system_prompt(self, user_profile: Dict) -> str:
        user_name = user_profile.get("name")
        name_context = f" User's name: {user_name}." if user_name else ""
    
        system_content = f"""You are MedicAI, an efficient assistant focused on completing instructions.{name_context}
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
    
        system_content = f"""You are MedicAI, friendly and welcoming.{greeting}
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
    
        system_content = f"""You are MedicAI, a friendly and engaging conversationalist.{name_context}{pref_context}
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
    user_id = "alice101" # Example user ID

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