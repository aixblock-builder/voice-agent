import chromadb
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from chromadb.config import Settings

class ChatHistoryManager:
    def __init__(self, persist_directory="./chroma_db_history"):
        """Initialize ChatHistoryManager with ChromaDB"""
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    allow_reset=True, 
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
        except Exception as e:
            print(f"⚠️ ChromaDB client creation failed, retrying: {e}")
            # If client creation fails, try with minimal settings
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection_name = "chat_history"
        self._init_collection()
    
    def _init_collection(self):
        """Initialize or get the chat history collection"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"📚 Connected to existing chat history collection")
        except Exception as e:
            try:
                # Create collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Chat conversation history storage"}
                )
                print(f"🆕 Created new chat history collection")
            except Exception as create_error:
                print(f"❌ Failed to create collection: {create_error}")
                # Last resort: try to get or create with minimal settings
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name
                    )
                    print(f"🔄 Using get_or_create_collection as fallback")
                except Exception as final_error:
                    print(f"💥 Complete ChromaDB failure: {final_error}")
                    raise
    
    def save_conversation_turn(self, session_id: str, user_message: str, 
                              bot_response: str, doc_files: List[str] = None, 
                              metadata: Dict[str, Any] = None) -> str:
        """
        Save a conversation turn (user message + bot response)
        
        Args:
            session_id: Unique session identifier
            user_message: User's input message
            bot_response: Bot's response
            doc_files: List of document files used
            metadata: Additional metadata
        
        Returns:
            turn_id: Unique identifier for this conversation turn
        """
        turn_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Prepare conversation turn data
        # Convert doc_files list to string for ChromaDB compatibility
        doc_files_str = ",".join(doc_files) if doc_files else ""
        
        conversation_data = {
            "session_id": session_id,
            "turn_id": turn_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": timestamp,
            "doc_files": doc_files_str,  # Store as comma-separated string
            "doc_files_list": json.dumps(doc_files or []),  # Store as JSON string for reconstruction
            "metadata": json.dumps(metadata or {})  # Convert metadata dict to JSON string
        }
        
        # Create document text for embedding (combine user message and response)
        document_text = f"User: {user_message}\nAssistant: {bot_response}"
        
        try:
            self.collection.add(
                documents=[document_text],
                metadatas=[conversation_data],
                ids=[turn_id]
            )
            print(f"💾 Saved conversation turn {turn_id} for session {session_id}")
            return turn_id
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
            return None
    
    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Get conversation history for a specific session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of turns to retrieve
        
        Returns:
            List of conversation turns
        """
        try:
            # Query by session_id in metadata
            results = self.collection.get(
                where={"session_id": session_id},
                limit=limit
            )
            
            if not results['metadatas']:
                return []
            
            # Sort by timestamp and reconstruct data
            history = []
            for metadata in results['metadatas']:
                # Reconstruct list/dict data from JSON strings
                reconstructed_metadata = metadata.copy()
                try:
                    reconstructed_metadata['doc_files'] = json.loads(metadata.get('doc_files_list', '[]'))
                    reconstructed_metadata['metadata'] = json.loads(metadata.get('metadata', '{}'))
                    # Remove the JSON string fields
                    reconstructed_metadata.pop('doc_files_list', None)
                except (json.JSONDecodeError, KeyError):
                    # Fallback for backward compatibility
                    reconstructed_metadata['doc_files'] = []
                    reconstructed_metadata['metadata'] = {}
                
                history.append(reconstructed_metadata)
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x['timestamp'], reverse=False)
            return history[-limit:]  # Return most recent conversations
            
        except Exception as e:
            print(f"❌ Error retrieving session history: {e}")
            return []
    
    def search_similar_conversations(self, query: str, session_id: str = None, 
                                   n_results: int = 5) -> List[Dict]:
        """
        Search for similar conversations using semantic search
        
        Args:
            query: Search query
            session_id: Optional session to filter by
            n_results: Number of results to return
        
        Returns:
            List of similar conversation turns
        """
        try:
            where_clause = {}
            if session_id:
                where_clause["session_id"] = session_id
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return []
            
            # Reconstruct data for search results
            reconstructed_results = []
            for metadata in results['metadatas'][0]:
                reconstructed_metadata = metadata.copy()
                try:
                    reconstructed_metadata['doc_files'] = json.loads(metadata.get('doc_files_list', '[]'))
                    reconstructed_metadata['metadata'] = json.loads(metadata.get('metadata', '{}'))
                    reconstructed_metadata.pop('doc_files_list', None)
                except (json.JSONDecodeError, KeyError):
                    reconstructed_metadata['doc_files'] = []
                    reconstructed_metadata['metadata'] = {}
                reconstructed_results.append(reconstructed_metadata)
            
            return reconstructed_results
            
        except Exception as e:
            print(f"❌ Error searching conversations: {e}")
            return []
    
    def format_history_for_context(self, history: List[Dict], max_turns: int = 5) -> str:
        """
        Format conversation history for use as context in prompts
        
        Args:
            history: List of conversation turns
            max_turns: Maximum number of turns to include
        
        Returns:
            Formatted history string
        """
        if not history:
            return ""
        
        # Take the most recent turns
        recent_history = history[-max_turns:] if len(history) > max_turns else history
        
        formatted_turns = []
        for turn in recent_history:
            user_msg = turn.get('user_message', '')
            bot_response = turn.get('bot_response', '')
            formatted_turns.append(f"User: {user_msg}\nAssistant: {bot_response}")
        
        return "\n\n--- Previous Conversation ---\n" + "\n\n".join(formatted_turns) + "\n--- End Previous Conversation ---\n"
    
    def create_new_session(self, title: str = None) -> Dict[str, str]:
        """
        Create a new session with optional title
        
        Args:
            title: Optional title for the session
            
        Returns:
            Dict with session_id and title
        """
        session_id = str(uuid.uuid4())
        session_title = title or f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Save session metadata as a special conversation turn
        try:
            session_metadata = {
                "session_id": session_id,
                "turn_id": f"session_meta_{session_id}",
                "user_message": f"SESSION_CREATED: {session_title}",
                "bot_response": "Session initialized",
                "timestamp": datetime.now().isoformat(),
                "doc_files": json.dumps([]),
                "metadata": json.dumps({
                    "is_session_metadata": True,
                    "session_title": session_title,
                    "created_at": datetime.now().isoformat()
                })
            }
            
            self.collection.add(
                documents=[f"Session: {session_title}"],
                metadatas=[session_metadata],
                ids=[f"session_meta_{session_id}"]
            )
            print(f"🆕 Created session {session_id} with title: {session_title}")
            
        except Exception as e:
            print(f"⚠️ Failed to save session metadata: {e}")
        
        return {"session_id": session_id, "title": session_title}
    
    def close_client(self):
        """Close ChromaDB client safely"""
        try:
            if hasattr(self, 'client'):
                del self.client
            print("🔐 ChromaDB client closed successfully")
        except Exception as e:
            print(f"⚠️ Error closing ChromaDB client: {e}")
    
    def get_all_sessions(self, limit: int = 100) -> List[Dict]:
        """
        Get list of all sessions with metadata
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of sessions with metadata
        """
        try:
            # Get all conversations
            results = self.collection.get(
                limit=limit * 10  # Get more to ensure we have enough unique sessions
            )
            
            if not results['metadatas']:
                return []
            
            # Group by session_id and collect metadata
            sessions_data = {}
            for metadata in results['metadatas']:
                session_id = metadata.get('session_id')
                if not session_id:
                    continue
                
                if session_id not in sessions_data:
                    # Try to get session title from metadata
                    session_title = "Untitled Session"
                    try:
                        meta_data = json.loads(metadata.get('metadata', '{}'))
                        if meta_data.get('is_session_metadata'):
                            session_title = meta_data.get('session_title', session_title)
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    
                    sessions_data[session_id] = {
                        'session_id': session_id,
                        'title': session_title,
                        'turn_count': 0,
                        'first_message': '',
                        'last_message': '',
                        'created_at': metadata.get('timestamp', ''),
                        'updated_at': metadata.get('timestamp', ''),
                        'doc_files_used': set()
                    }
                
                # Update session data (skip session metadata from turn count)
                session_info = sessions_data[session_id]
                try:
                    meta_data = json.loads(metadata.get('metadata', '{}'))
                    if not meta_data.get('is_session_metadata', False):
                        session_info['turn_count'] += 1
                except (json.JSONDecodeError, AttributeError):
                    session_info['turn_count'] += 1
                
                # Track first and last messages
                timestamp = metadata.get('timestamp', '')
                if timestamp < session_info['created_at'] or not session_info['created_at']:
                    session_info['created_at'] = timestamp
                    session_info['first_message'] = metadata.get('user_message', '')[:100] + '...' if len(metadata.get('user_message', '')) > 100 else metadata.get('user_message', '')
                
                if timestamp > session_info['updated_at'] or not session_info['updated_at']:
                    session_info['updated_at'] = timestamp
                    session_info['last_message'] = metadata.get('user_message', '')[:100] + '...' if len(metadata.get('user_message', '')) > 100 else metadata.get('user_message', '')
                
                # Track doc files
                try:
                    doc_files = json.loads(metadata.get('doc_files_list', '[]'))
                    session_info['doc_files_used'].update(doc_files)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Convert to list and format
            sessions_list = []
            for session_info in sessions_data.values():
                session_info['doc_files_used'] = list(session_info['doc_files_used'])
                sessions_list.append(session_info)
            
            # Sort by updated_at (most recent first)
            sessions_list.sort(key=lambda x: x['updated_at'], reverse=True)
            
            return sessions_list[:limit]
            
        except Exception as e:
            print(f"❌ Error retrieving sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all conversations for a session"""
        try:
            # Get all turns for the session
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"🗑️ Deleted {len(results['ids'])} turns for session {session_id}")
                return True
            
            return False
        except Exception as e:
            print(f"❌ Error deleting session: {e}")
            return False 