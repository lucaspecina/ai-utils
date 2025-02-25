#!/usr/bin/env python3
"""
Memory Manager Component

This module handles:
- Short-term conversation memory
- Long-term vector memory using FAISS
- User preferences storage and retrieval
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_manager.log"),
    ]
)
logger = logging.getLogger("memory_manager")

# Try to import required libraries
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install faiss-cpu sentence-transformers")
    sys.exit(1)

class MemoryManager:
    """
    Manages different types of memory for the assistant.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize memory components."""
        self.config = config or {
            "short_term_limit": 10,  # conversations
            "vector_db_path": "data/memory/vector_store",
            "user_preferences_path": "data/memory/user_prefs.json",
        }
        
        # Initialize various memory components
        self.short_term_memory = []  # Simple list for recent conversations
        self.user_preferences = self._load_user_preferences()
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        print(f"Memory Manager initialized. Data stored in: {os.path.dirname(self.config['user_preferences_path'])}")
    
    def _initialize_vector_store(self):
        """Initialize vector database for long-term memory using FAISS."""
        try:
            logger.info("Initializing vector store for long-term memory")
            print("Initializing vector store for long-term memory...")
            
            # Create directories
            vector_db_path = self.config["vector_db_path"]
            os.makedirs(vector_db_path, exist_ok=True)
            
            # Initialize sentence transformer for embeddings
            print("  • Loading sentence transformer model...")
            model_name = "all-MiniLM-L6-v2"  # Good balance of quality and speed
            self.embedding_model = SentenceTransformer(model_name)
            print("  • Sentence transformer loaded successfully")
            
            # Check if we have existing index
            index_path = f"{vector_db_path}/faiss_index.bin"
            texts_path = f"{vector_db_path}/texts.json"
            
            if os.path.exists(index_path) and os.path.exists(texts_path):
                # Load existing index
                print("  • Loading existing vector indices...")
                logger.info("Loading existing vector store")
                self.index = faiss.read_index(index_path)
                
                with open(texts_path, 'r') as f:
                    self.stored_texts = json.load(f)
                print(f"  • Loaded existing vector store with {len(self.stored_texts)} entries")
            else:
                # Create new index - using L2 distance
                print("  • Creating new vector indices...")
                logger.info("Creating new vector store")
                dimension = self.embedding_model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(dimension)
                self.stored_texts = []
                
                # Add initial entry
                print("  • Adding initial entry to vector store...")
                self._add_texts_to_index(["Initial memory entry for the assistant."], 
                                       [{"source": "initialization"}])
                print("  • New vector store created successfully")
            
            logger.info(f"Vector store initialized with {len(self.stored_texts)} entries")
            return {
                "index": self.index,
                "texts": self.stored_texts,
                "embedding_model": self.embedding_model
            }
            
        except Exception as e:
            print(f"ERROR initializing vector store: {e}")
            logger.error(f"Error initializing vector store: {e}")
            logger.error("Falling back to simple list-based storage")
            return {"texts": [], "metadata": []}
    
    def _add_texts_to_index(self, texts, metadatas=None):
        """Add texts to the FAISS index."""
        if not metadatas:
            metadatas = [{}] * len(texts)
            
        # Create embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store the texts and metadata
        for text, metadata in zip(texts, metadatas):
            self.stored_texts.append({
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Save the index and texts
        vector_db_path = self.config["vector_db_path"]
        faiss.write_index(self.index, f"{vector_db_path}/faiss_index.bin")
        
        with open(f"{vector_db_path}/texts.json", 'w') as f:
            json.dump(self.stored_texts, f)
    
    def _load_user_preferences(self) -> Dict:
        """Load user preferences from storage."""
        prefs_path = self.config["user_preferences_path"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(prefs_path), exist_ok=True)
        
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading user preferences: {e}")
                return {}
        else:
            # Initialize with empty preferences
            return {}
    
    def add_to_short_term_memory(self, user_input: str, assistant_response: str):
        """Add an interaction to short-term memory."""
        self.short_term_memory.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Limit size of short-term memory
        max_items = self.config["short_term_limit"]
        if len(self.short_term_memory) > max_items:
            self.short_term_memory = self.short_term_memory[-max_items:]
        
        print(f"Added to short-term memory. Current size: {len(self.short_term_memory)}")
    
    def add_to_long_term_memory(self, text: str, metadata: Dict = None):
        """Add information to long-term vector memory."""
        try:
            if isinstance(self.vector_store, dict) and "embedding_model" in self.vector_store:
                # Using FAISS
                self._add_texts_to_index([text], [metadata or {}])
                print(f"Added to long-term memory: {text[:50]}...")
            else:
                # Fallback for simple storage
                if "texts" in self.vector_store:
                    self.vector_store["texts"].append(text)
                    if "metadata" in self.vector_store:
                        self.vector_store["metadata"].append(metadata or {})
                print(f"Added to long-term memory (simple): {text[:50]}...")
        except Exception as e:
            logger.error(f"Error adding to long-term memory: {e}")
            print(f"Error adding to long-term memory: {e}")
    
    def query_long_term_memory(self, query: str, k: int = 5) -> List[Dict]:
        """Query long-term memory for relevant information."""
        try:
            if isinstance(self.vector_store, dict) and "embedding_model" in self.vector_store:
                # Using FAISS for semantic search
                query_embedding = self.vector_store["embedding_model"].encode([query])
                distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
                
                results = []
                for idx in indices[0]:
                    if idx != -1 and idx < len(self.stored_texts):  # -1 indicates no match
                        results.append(self.stored_texts[idx])
                
                print(f"Found {len(results)} relevant memories for query: {query[:30]}...")
                return results
            else:
                # Fallback for simple storage - just return most recent entries
                if "texts" in self.vector_store:
                    results = []
                    for i in range(min(k, len(self.vector_store["texts"]))):
                        text = self.vector_store["texts"][-(i+1)]
                        metadata = self.vector_store["metadata"][-(i+1)] if "metadata" in self.vector_store else {}
                        results.append({"text": text, "metadata": metadata})
                    return results
                return []
        except Exception as e:
            logger.error(f"Error querying long-term memory: {e}")
            print(f"Error querying long-term memory: {e}")
            return []
    
    def update_user_preferences(self, key: str, value: Any):
        """Update user preferences with new information."""
        self.user_preferences[key] = value
        
        # Save to disk
        prefs_path = self.config["user_preferences_path"]
        os.makedirs(os.path.dirname(prefs_path), exist_ok=True)
        
        try:
            with open(prefs_path, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
            print(f"Updated user preference: {key} = {value}")
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
            print(f"Error saving user preferences: {e}")
    
    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """Get recent conversation history."""
        if limit is None:
            return self.short_term_memory
        return self.short_term_memory[-limit:]
    
    def get_user_preferences(self) -> Dict:
        """Get all user preferences."""
        return self.user_preferences
    
    def infer_user_preferences(self, interaction_data: Dict):
        """
        Analyze interactions to infer user preferences automatically.
        """
        if "user" in interaction_data:
            text = interaction_data["user"].lower()
            
            # Example: Detect time preferences
            if "morning" in text or "early" in text:
                self.update_user_preferences("preferred_time", "morning")
            elif "evening" in text or "night" in text:
                self.update_user_preferences("preferred_time", "evening")
                
            # Example: Detect communication style preferences
            if "brief" in text or "short" in text:
                self.update_user_preferences("communication_style", "concise")
            elif "detail" in text or "explain" in text:
                self.update_user_preferences("communication_style", "detailed")


# Test function to run this module independently
def test_memory_manager():
    """Test the memory manager functionality."""
    print("Testing Memory Manager...")
    
    # Create a test configuration
    test_config = {
        "short_term_limit": 5,
        "vector_db_path": "data/memory/vector_store",
        "user_preferences_path": "data/memory/user_prefs.json",
    }
    
    # Create memory manager
    memory = MemoryManager(test_config)
    
    # Test short-term memory
    print("\nTesting short-term memory...")
    memory.add_to_short_term_memory("What's the weather today?", "It's sunny and 75°F.")
    memory.add_to_short_term_memory("Set a reminder for tomorrow", "I've set a reminder for tomorrow.")
    
    history = memory.get_conversation_history()
    print(f"Conversation history ({len(history)} items):")
    for item in history:
        print(f"  User: {item['user']}")
        print(f"  Assistant: {item['assistant']}")
        print(f"  Time: {item['timestamp']}")
    
    # Test user preferences
    print("\nTesting user preferences...")
    memory.update_user_preferences("favorite_color", "blue")
    memory.update_user_preferences("preferred_language", "English")
    
    prefs = memory.get_user_preferences()
    print("User preferences:")
    for key, value in prefs.items():
        print(f"  {key}: {value}")
    
    # Test long-term memory
    print("\nTesting long-term memory...")
    memory.add_to_long_term_memory(
        "User likes to go hiking on weekends and prefers mountain trails.",
        {"category": "user_hobby", "confidence": "high"}
    )
    memory.add_to_long_term_memory(
        "User mentioned they have a dog named Max who is a Golden Retriever.",
        {"category": "user_pet", "confidence": "high"}
    )
    
    # Test querying
    print("\nTesting memory queries...")
    queries = [
        "What does the user like to do on weekends?",
        "Does the user have any pets?",
        "What's the user's favorite color?"
    ]
    
    for query in queries:
        results = memory.query_long_term_memory(query, k=2)
        print(f"\nQuery: {query}")
        print(f"Results ({len(results)}):")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result['text'][:100]}...")
            if "metadata" in result:
                print(f"  Metadata: {result['metadata']}")
    
    print("\nMemory Manager test completed")


if __name__ == "__main__":
    test_memory_manager() 