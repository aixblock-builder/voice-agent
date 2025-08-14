import asyncio
import json
import os
from typing import Any, Dict, Iterator

import gradio as gr
import spaces
import torch
from aixblock_ml.model import AIxBlockMLBase
from huggingface_hub import HfFolder, login
from loguru import logger
from mcp.server.fastmcp import FastMCP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import gc
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_community.vectorstores import Chroma, Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
from datetime import datetime
import boto3
import ftplib
import paramiko
from smb.SMBConnection import SMBConnection
from typing import List, Dict, Any
import hashlib
import io
import tempfile
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import re
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.schema import BaseOutputParser
import redis

# ------------------------------------------------------------------------------
HfFolder.save_token("hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")
login(token = "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")

CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(os.environ["CUDA_VISIBLE_DEVICES"])


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}
# Parameters for model demo
model_demo = None
tokenizer_demo = None
model_loaded_demo = False
# Parameters for model deployment
pipe_prediction = None
tokenizer = None
model_predict = None

# 🔧 MCP Tools Registry
MCP_TOOLS_REGISTRY = {}
REMOTE_MCP_CONNECTIONS = {}
TOOLS = []

KNOWLEDGE_BASES = {}  # Store active knowledge base connections

# 📝 Prompt Templates Registry
CUSTOM_TEMPLATES = {
    "system": {
        "template": """You are an AI assistant.

Instructions:
- Use knowledge base information when relevant
- For tool usage, respond with: TOOL_CALL:tool_name:{{"param": "value"}}
- Always provide helpful, accurate responses""",
        "input_variables": ["kb_context", "has_tools"],
        "description": "Default system message template"
    },
    "tool_detection": {
        "template": "Query: {query}\nResponse:",
        "input_variables": ["query"],
        "description": "Simple tool detection template"
    },
    "context_assembly": {
        "template": """You are an AI assistant.

Instructions:
- Use knowledge base information when relevant
- Always provide helpful, accurate responses

CONVERSATION HISTORY:
{history}

KNOWLEDGE BASE:
{kb_context}""",
        "input_variables": ["history", "kb_context"],
        "description": "Context assembly template"
    }
}

class PromptTemplateManager:
    """Centralized prompt template management with dynamic updates"""
    
    @staticmethod
    def get_template(template_name: str) -> PromptTemplate:
        """Get template by name"""
        if template_name not in CUSTOM_TEMPLATES:
            return None
        
        print(CUSTOM_TEMPLATES)
        
        config = CUSTOM_TEMPLATES[template_name]

        return PromptTemplate(
            input_variables=config["input_variables"],
            template=config["template"]
        )
    
    @staticmethod
    def get_system_template():
        """System message template with KB and tools context"""
        template = PromptTemplateManager.get_template("system")
        if template:
            return template
        # Fallback default
        return PromptTemplate(
            input_variables=["kb_context", "has_tools"],
            template="""You are an AI assistant{tools_info}.

{kb_section}

Instructions:
- Use knowledge base information when relevant
- For tool usage, respond with: TOOL_CALL:tool_name:{{"param": "value"}}
- Always provide helpful, accurate responses""",
            partial_variables={
                "tools_info": " with access to external tools" if MCP_TOOLS_REGISTRY else "",
                "kb_section": "{kb_context}" if "{kb_context}" else "",
            }
        )
    
    @staticmethod
    def get_context_assembly_template(template_name = "context_assembly"):
        """Template for combining multiple context sources"""
        template = PromptTemplateManager.get_template(template_name)
        if template:
            return template
        # Fallback default    
        return PromptTemplate(
            input_variables=["history", "kb_context"],
            template="""CONVERSATION HISTORY:
{history}

KNOWLEDGE BASE:
{kb_context}

"""
        )
    
    @staticmethod
    def update_template(template_name: str, template_text: str, input_variables: list, description: str = ""):
        """Update or create a custom template"""
        CUSTOM_TEMPLATES[template_name] = {
            "template": template_text,
            "input_variables": input_variables,
            "description": description,
            "updated_at": str(datetime.now())
        }
        return {"success": True, "message": f"Template '{template_name}' updated"}
    
    @staticmethod
    def list_templates():
        """List all available templates"""
        return {
            "templates": {name: {
                "description": config.get("description", ""),
                "variables": config["input_variables"],
                "updated_at": config.get("updated_at", "system_default")
            } for name, config in CUSTOM_TEMPLATES.items()},
            "count": len(CUSTOM_TEMPLATES)
        }
    
    @staticmethod
    def delete_template(template_name: str):
        """Delete a custom template"""
        if template_name not in CUSTOM_TEMPLATES:
            return {"error": f"Template '{template_name}' not found"}
        
        # Don't delete system templates
        if template_name in ["system", "tool_detection", "context_assembly"]:
            return {"error": "Cannot delete system templates"}
        
        del CUSTOM_TEMPLATES[template_name]
        return {"success": True, "message": f"Template '{template_name}' deleted"}
    
    @staticmethod
    def validate_template(template_text: str, input_variables: list):
        """Validate template syntax"""
        try:
            test_template = PromptTemplate(
                input_variables=input_variables,
                template=template_text
            )
            # Test format with dummy values
            test_values = {var: f"test_{var}" for var in input_variables}
            test_template.format(**test_values)
            return {"valid": True, "message": "Template is valid"}
        except Exception as e:
            return {"valid": False, "message": f"Template validation failed: {str(e)}"}

class KnowledgeBaseManager:
    """Manage knowledge base connections and queries"""
    
    @staticmethod
    def register_connections(name: str, memory_conn: dict = None, storage_conn: dict = None):
        """Register memory and storage connections for a knowledge base"""
        if not memory_conn and not storage_conn:
            return {"error": "At least one connection required"}
        
        kb_config = {
            "name": name,
            "memory": memory_conn,
            "storage": storage_conn,
            "vectorstore": None,
            "embeddings": None,
            "memory_db": None,
            "redis_client": None
        }
        
        # Initialize embeddings
        try:
            kb_config["embeddings"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize memory connection
            if memory_conn:
                db_type = memory_conn.get("databaseType", "").upper()
                
                if db_type == "POSTGRES":
                    # PostgreSQL connection
                    try:
                        conn = psycopg2.connect(
                            host=memory_conn["postgresHost"],
                            port=memory_conn["postgresPort"],
                            user=memory_conn["postgresUsername"],
                            password=memory_conn["postgresPassword"],
                            database=memory_conn["postgresDatabaseName"]
                        )
                        kb_config["memory_db"] = conn
                        # Use Chroma with Postgres backend (simplified)
                        persist_dir = f"./chroma_kb_{hashlib.md5(name.encode()).hexdigest()[:8]}"
                        kb_config["vectorstore"] = Chroma(
                            persist_directory=persist_dir,
                            embedding_function=kb_config["embeddings"]
                        )
                    except Exception as e:
                        logger.error(f"Postgres connection failed: {e}")
                
                elif db_type == "QDRANT":
                    # Qdrant vector database
                    try:
                        from qdrant_client import QdrantClient
                        qdrant_url = memory_conn.get("qdrantUrl")
                        qdrant_key = memory_conn.get("qdrantKey")
                        
                        if qdrant_url:
                            client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                            kb_config["vectorstore"] = Qdrant(
                                client=client,
                                collection_name=f"kb_{name}",
                                embeddings=kb_config["embeddings"]
                            )
                    except Exception as e:
                        logger.error(f"Qdrant connection failed: {e}")
                
                # Redis connection
                redis_url = memory_conn.get("redisUrl")
                if redis_url:
                    try:
                        kb_config["redis_client"] = redis.from_url(redis_url)
                    except Exception as e:
                        logger.error(f"Redis connection failed: {e}")
            
            # Fallback to local Chroma if no vector store initialized
            if not kb_config["vectorstore"]:
                persist_dir = f"./chroma_kb_{hashlib.md5(name.encode()).hexdigest()[:8]}"
                kb_config["vectorstore"] = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=kb_config["embeddings"]
                )
            
            KNOWLEDGE_BASES[name] = kb_config
            return {"message": f"Knowledge base '{name}' registered successfully"}
            
        except Exception as e:
            return {"error": f"Failed to initialize knowledge base: {str(e)}"}
    
    @staticmethod
    def load_documents_from_storage(name: str, query: str = "") -> List[str]:
        """Load documents from all supported storage types"""
        if name not in KNOWLEDGE_BASES:
            return []
        
        kb = KNOWLEDGE_BASES[name]
        storage_conn = kb.get("storage")
        
        if not storage_conn:
            return []
        
        documents = []
        db_type = storage_conn.get("databaseType", "").upper()
        
        try:
            if db_type == "S3":
                documents.extend(KnowledgeBaseManager._load_from_s3(storage_conn))
            elif db_type == "FTP":
                documents.extend(KnowledgeBaseManager._load_from_ftp(storage_conn))
            elif db_type == "SFTP":
                documents.extend(KnowledgeBaseManager._load_from_sftp(storage_conn))
            elif db_type == "SMB2":
                documents.extend(KnowledgeBaseManager._load_from_smb2(storage_conn))
            
        except Exception as e:
            logger.error(f"Storage load failed for {db_type}: {e}")
        
        return documents
    
    @staticmethod
    def _load_from_s3(storage_conn: dict) -> List[str]:
        """Load documents from S3"""
        documents = []
        s3_client = boto3.client(
            's3',
            endpoint_url=storage_conn["s3Endpoint"],
            aws_access_key_id=storage_conn["s3AccessKey"],
            aws_secret_access_key=storage_conn["s3SecretKey"],
            region_name=storage_conn["s3Region"]
        )
        
        bucket = storage_conn["s3BucketName"]
        response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=50)
        
        import zipfile
        import xml.etree.ElementTree as ET
        import io

        def extract_text_from_docx(docx_content):
            """Extract text from DOCX file without external libraries"""
            try:
                with zipfile.ZipFile(io.BytesIO(docx_content)) as zip_file:
                    # Read the main document XML
                    xml_content = zip_file.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    # Extract all text nodes
                    text_parts = []
                    for elem in root.iter():
                        if elem.text:
                            text_parts.append(elem.text)
                    
                    return '\n'.join(text_parts)
            except Exception as e:
                return f"Error reading DOCX: {e}"

        for obj in response.get('Contents', []):
            if obj['Key'].endswith(('.txt', '.md', '.json', '.pdf', '.docx')):
                try:
                    file_obj = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                    
                    if obj['Key'].endswith('.docx'):
                        # Xử lý file DOCX bằng cách extract từ ZIP
                        doc_content = file_obj['Body'].read()
                        content = extract_text_from_docx(doc_content)
                    else:
                        # Xử lý các file text khác
                        content = file_obj['Body'].read().decode('utf-8', errors='ignore')
                    
                    documents.append(f"File: {obj['Key']}\n{content}")
                except Exception as e:
                    logger.error(f"S3 file load failed {obj['Key']}: {e}")

        return documents
    
    @staticmethod
    def _load_from_ftp(storage_conn: dict) -> List[str]:
        """Load documents from FTP"""
        documents = []
        try:
            ftp = ftplib.FTP()
            ftp.connect(storage_conn["ftpHost"], int(storage_conn.get("ftpPort", 21)))
            ftp.login(storage_conn["ftpUsername"], storage_conn["ftpPassword"])
            
            files = ftp.nlst()
            for filename in files[:10]:  # Limit files
                if filename.endswith(('.txt', '.md', '.json')):
                    try:
                        with io.BytesIO() as bio:
                            ftp.retrbinary(f'RETR {filename}', bio.write)
                            content = bio.getvalue().decode('utf-8', errors='ignore')
                            documents.append(f"File: {filename}\n{content}")
                    except Exception as e:
                        logger.error(f"FTP file load failed {filename}: {e}")
            
            ftp.quit()
        except Exception as e:
            logger.error(f"FTP connection failed: {e}")
        
        return documents
    
    @staticmethod
    def _load_from_sftp(storage_conn: dict) -> List[str]:
        """Load documents from SFTP"""
        documents = []
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                storage_conn["sftpHost"],
                port=int(storage_conn.get("sftpPort", 22)),
                username=storage_conn["sftpUsername"],
                password=storage_conn["sftpPassword"]
            )
            
            sftp = ssh.open_sftp()
            files = sftp.listdir('.')
            
            for filename in files[:10]:  # Limit files
                if filename.endswith(('.txt', '.md', '.json')):
                    try:
                        with sftp.open(filename, 'r') as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            documents.append(f"File: {filename}\n{content}")
                    except Exception as e:
                        logger.error(f"SFTP file load failed {filename}: {e}")
            
            sftp.close()
            ssh.close()
        except Exception as e:
            logger.error(f"SFTP connection failed: {e}")
        
        return documents
    
    @staticmethod
    def _load_from_smb2(storage_conn: dict) -> List[str]:
        """Load documents from SMB2/CIFS"""
        documents = []
        try:
            conn = SMBConnection(
                storage_conn["smb2Username"],
                storage_conn["smb2Password"],
                "python-client",
                storage_conn["smb2Host"],
                domain=storage_conn.get("smb2Domain", ""),
                use_ntlm_v2=True
            )
            
            if conn.connect(storage_conn["smb2Host"], int(storage_conn.get("smb2Port", 445))):
                share_name = storage_conn["smb2Share"]
                files = conn.listPath(share_name, '/')
                
                for file_info in files[:10]:  # Limit files
                    if file_info.filename.endswith(('.txt', '.md', '.json')):
                        try:
                            with tempfile.NamedTemporaryFile() as tmp_file:
                                conn.retrieveFile(share_name, file_info.filename, tmp_file)
                                tmp_file.seek(0)
                                content = tmp_file.read().decode('utf-8', errors='ignore')
                                documents.append(f"File: {file_info.filename}\n{content}")
                        except Exception as e:
                            logger.error(f"SMB2 file load failed {file_info.filename}: {e}")
                
                conn.close()
        except Exception as e:
            logger.error(f"SMB2 connection failed: {e}")
        
        return documents
    
    @staticmethod
    def search_knowledge(name: str, query: str, limit: int = 5) -> str:
        """Search knowledge base for relevant information"""
        if name not in KNOWLEDGE_BASES:
            return "Knowledge base not found"
        
        kb = KNOWLEDGE_BASES[name]
        vectorstore = kb.get("vectorstore")
        
        if not vectorstore:
            return "Vector store not initialized"
        
        try:
            # Get similar documents
            docs = vectorstore.similarity_search(query, k=limit)
            if not docs:
                return "No relevant information found"
            
            # Combine results
            context = "\n---\n".join([doc.page_content for doc in docs])
            return f"Relevant information:\n{context}"
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    @staticmethod
    def search_all_knowledge_bases(query: str, limit: int = 3) -> str:
        """Search all available knowledge bases for relevant information"""
        if not KNOWLEDGE_BASES:
            return ""
        
        all_context = []
        
        for kb_name, kb_config in KNOWLEDGE_BASES.items():
            vectorstore = kb_config.get("vectorstore")
            if not vectorstore:
                continue
                
            try:
                docs = vectorstore.similarity_search(query, k=limit)
                if docs:
                    # kb_context = f"[From {kb_name}]:\n" + "\n".join([doc.page_content for doc in docs])
                    kb_context = f"\n" + "\n".join([doc.page_content for doc in docs])
                    all_context.append(kb_context)
            except Exception as e:
                logger.error(f"KB search failed for {kb_name}: {e}")
                continue
        
        return "\n\n---\n\n".join(all_context) if all_context else ""
    
    @staticmethod
    def update_knowledge_base(name: str):
        """Update knowledge base with latest documents from storage"""
        if name not in KNOWLEDGE_BASES:
            return {"error": "Knowledge base not found"}
        
        try:
            # Load documents from storage
            documents = KnowledgeBaseManager.load_documents_from_storage(name)
            
            if not documents:
                return {"message": "No documents to update"}
            
            # Split and add to vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            kb = KNOWLEDGE_BASES[name]
            vectorstore = kb["vectorstore"]
            
            for doc_content in documents:
                chunks = text_splitter.split_text(doc_content)
                vectorstore.add_texts(chunks)
            
            return {"message": f"Updated with {len(documents)} documents"}
            
        except Exception as e:
            return {"error": f"Update failed: {str(e)}"}

class ToolCall(BaseModel):
    """Tool call model for parsing"""
    tool_name: Optional[str] = Field(None, description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    confidence: float = Field(0.0, description="Confidence score")

class AIToolExtractor:
    """AI-based tool extraction using sentence similarity"""
    
    def __init__(self):
        self.embeddings_model = None
        self.vectorizer = None
        self.use_tfidf = False
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer for similarity matching"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use lightweight sentence transformer
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Loaded SentenceTransformer model")
        except ImportError:
            try:
                # Fallback to sklearn TF-IDF if sentence-transformers not available
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                self.use_tfidf = True
                print("✅ Using TF-IDF fallback")
            except ImportError:
                print("⚠️ No ML libraries available, using rule-based matching")
                self.embeddings_model = None
                self.use_tfidf = False
    
    def extract_tool_info(self, text: str, available_tools: Dict[str, Any]) -> tuple:
        """Extract tool name and parameters using similarity matching"""
        if not available_tools:
            return None, {}, 0.0
        
        # Prepare tool descriptions for similarity matching
        tool_texts = []
        tool_names = []
        
        for tool_name, info in available_tools.items():
            tool_desc = f"{tool_name} {info.get('description', '')}"
            tool_texts.append(tool_desc)
            tool_names.append(tool_name)
        
        # Find best matching tool
        best_tool, confidence = self._find_best_match(text, tool_texts, tool_names)
        print("===best_tool===", best_tool)
        print("===confidence===", confidence)
        
        params = self._extract_params_with_ner(text, available_tools[best_tool])
        return best_tool, params, confidence

    
    def _find_best_match(self, query: str, tool_texts: list, tool_names: list) -> tuple:
        """Find best matching tool using embeddings or TF-IDF"""
        if self.embeddings_model:
            # Use sentence transformers
            try:
                query_embedding = self.embeddings_model.encode([query])
                tool_embeddings = self.embeddings_model.encode(tool_texts)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
                best_idx = similarities.argmax()
                confidence = float(similarities[best_idx])
                
                return tool_names[best_idx], confidence
            except Exception as e:
                print(f"SentenceTransformer failed: {e}")
                
        if self.use_tfidf and self.vectorizer:
            # Use TF-IDF fallback
            try:
                all_texts = tool_texts + [query]
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
                best_idx = similarities.argmax()
                confidence = float(similarities[best_idx])
                
                return tool_names[best_idx], confidence
            except Exception as e:
                print(f"TF-IDF failed: {e}")
        
        # Rule-based fallback
        best_tool = None
        best_score = 0
        
        query_lower = query.lower()
        for i, tool_name in enumerate(tool_names):
            score = 0
            # Direct name match
            if tool_name.lower() in query_lower:
                score += 0.8
            # Partial name match
            tool_parts = tool_name.lower().split('.')
            if any(part in query_lower for part in tool_parts):
                score += 0.5
            # Description word match
            tool_desc = tool_texts[i].lower()
            common_words = set(query_lower.split()) & set(tool_desc.split())
            score += len(common_words) * 0.1
            
            if score > best_score:
                best_score = score
                best_tool = tool_name
        
        return best_tool, min(best_score, 0.9)
    
    def _extract_params_with_ner(self, text: str, tool_info: Dict) -> Dict[str, Any]:
        """Extract parameters using AI model"""
        schema = tool_info.get('parameters', {})
        print("===schema===", schema)
        return self._ai_extract_params(text, schema)
    
    def _ai_extract_params(self, text: str, schema: Dict) -> Dict[str, Any]:
        """Use AI to extract all parameters from text"""
        params = {}
        print("===text===", text)
        print("===schema===", schema)
        
        if not schema:
            return params
        
        try:
            # First try direct regex extraction for all parameters at once
            params = self._extract_all_params_regex(text, schema)
            
            # If AI model is available, enhance extraction for missing parameters
            if self.embeddings_model and len(params) < len(schema):
                missing_params = set(schema.keys()) - set(params.keys())
                
                for param_name in missing_params:
                    param_info = schema[param_name]
                    
                    # Create context-aware queries
                    queries = [
                        f"{param_name}",
                        f"parameter {param_name}",
                        f"{param_name} value",
                        f"set {param_name}"
                    ]
                    
                    # Split text into meaningful chunks
                    chunks = self._create_text_chunks(text, param_name)
                    
                    best_value = None
                    best_score = 0
                    
                    for query in queries:
                        if not chunks:
                            continue
                            
                        query_embedding = self.embeddings_model.encode([query])
                        chunk_embeddings = self.embeddings_model.encode(chunks)
                        
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                        
                        max_idx = similarities.argmax()
                        if similarities[max_idx] > best_score:
                            best_score = similarities[max_idx]
                            value = self._extract_value_from_chunk(chunks[max_idx], param_name, param_info)
                            if value is not None:
                                best_value = value
                    
                    if best_value is not None and best_score > 0.1:
                        params[param_name] = best_value
                            
        except Exception as e:
            print(f"AI param extraction failed: {e}")
            # Fallback to regex-only extraction
            params = self._extract_all_params_regex(text, schema)
            
        return params

    def _extract_value_from_chunk(self, chunk: str, param_name: str, param_info: Dict) -> any:
        """Extract parameter value from text chunk using enhanced pattern matching"""
        chunk_lower = chunk.lower()
        param_lower = param_name.lower()
        
        import re
        
        # Enhanced patterns with better capturing groups and flexibility
        patterns = [
            # Direct colon patterns: "param: value" or "param:value"
            rf"{re.escape(param_lower)}\s*:\s*([^,\s]+(?:\.\d+)?)",
            rf"{re.escape(param_lower)}\s*:\s*([^,]+?)(?:\s*,\s*\w+\s*[:=]|\s*$)",
            rf"{re.escape(param_lower)}\s*:\s*'([^']*)'",  # quoted values
            rf"{re.escape(param_lower)}\s*:\s*\"([^\"]*)\"",  # double quoted
            
            # Equals patterns: "param = value" or "param=value" 
            rf"{re.escape(param_lower)}\s*=\s*([^,\s]+(?:\.\d+)?)",
            rf"{re.escape(param_lower)}\s*=\s*([^,]+?)(?:\s*,\s*\w+\s*[:=]|\s*$)",
            
            # Natural language patterns
            rf"{re.escape(param_lower)}\s+(?:is\s+)?set\s+to\s+([^,\s]+(?:\.\d+)?)",
            rf"{re.escape(param_lower)}\s+(?:is\s+)?([^,\s]+(?:\.\d+)?)(?:\s*,|\s*and|\s*$)",
            
            # "with param value" patterns
            rf"with\s+{re.escape(param_lower)}\s+([^,\s]+(?:\.\d+)?)",
            rf"with\s+{re.escape(param_lower)}\s*:\s*([^,\s]+(?:\.\d+)?)",
            
            # Value before param patterns: "value param"  
            rf"([^,\s]+(?:\.\d+)?)\s+{re.escape(param_lower)}(?:\s*,|\s*and|\s*$)",
            
            # Quoted values
            rf"{re.escape(param_lower)}\s*[:=]\s*['\"]([^'\"]+)['\"]",
            rf"['\"]([^'\"]*{re.escape(param_lower)}[^'\"]*)['\"]",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, chunk_lower)
            for match in matches:
                # Get the actual text from original chunk (preserving case)
                start_pos = match.start(1) 
                end_pos = match.end(1)
                
                # Map back to original chunk to preserve case
                value = chunk[start_pos:end_pos].strip().strip("'\"").strip()
                
                if value and not value.isspace():
                    converted = self._convert_param_type(value, param_info)
                    if converted is not None:
                        return converted
        
        # Special handling for numeric values that might be standalone
        if param_info.get('type') in ['integer', 'number', 'float']:
            # Look for standalone numbers near the parameter name
            words = chunk.split()
            param_words = param_lower.split()
            
            for i, word in enumerate(words):
                if any(pw in word.lower() for pw in param_words):
                    # Check surrounding words for numbers
                    for j in range(max(0, i-2), min(len(words), i+3)):
                        if j != i:
                            number_match = re.match(r'^-?\d*\.?\d+$', words[j].strip(',:'))
                            if number_match:
                                converted = self._convert_param_type(number_match.group(), param_info)
                                if converted is not None:
                                    return converted
        
        return None

    def _extract_all_params_regex(self, text: str, schema: Dict) -> Dict[str, Any]:
        """Extract all parameters using regex patterns - helper method"""
        params = {}
        text_lower = text.lower()
        
        import re
        
        for param_name, param_info in schema.items():
            param_lower = param_name.lower()
            
        # Comprehensive pattern list for this parameter
            patterns = [
                rf"{re.escape(param_lower)}\s*:\s*'([^']*)'",  # quoted values first
                rf"{re.escape(param_lower)}\s*:\s*\"([^\"]*)\"",
                rf"{re.escape(param_lower)}\s*:\s*([^,]+?)(?:\s*,\s*\w+\s*[:=]|\s*$)",  # long values
                rf"{re.escape(param_lower)}\s*:\s*([^,\s]+(?:\.\d+)?)",  # short values
                rf"{re.escape(param_lower)}\s*=\s*([^,\s]+(?:\.\d+)?)",
                rf"with\s+{re.escape(param_lower)}\s+([^,\s]+(?:\.\d+)?)",
                rf"{re.escape(param_lower)}\s+set\s+to\s+([^,\s]+(?:\.\d+)?)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Get actual value from original text
                    start_pos = match.start(1)
                    end_pos = match.end(1)
                    value = text[start_pos:end_pos].strip().strip("'\"")
                    
                    converted = self._convert_param_type(value, param_info)
                    if converted is not None:
                        params[param_name] = converted
                        break
        
        return params

    def _create_text_chunks(self, text: str, param_name: str) -> list:
        """Create meaningful text chunks for parameter extraction"""
        chunks = []
        
        # Split by common delimiters
        for delimiter in [',', ';', ' and ', ' with ']:
            parts = text.split(delimiter)
            chunks.extend([part.strip() for part in parts if part.strip()])
        
        # Add sentences containing the parameter name
        sentences = [s.strip() for s in text.replace(',', '.').split('.') if s.strip()]
        param_sentences = [s for s in sentences if param_name.lower() in s.lower()]
        chunks.extend(param_sentences)
        
        # Add the full text as fallback
        chunks.append(text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)
        
        return unique_chunks

    def _convert_param_type(self, value: str, param_info: Dict) -> any:
        """Convert string value to appropriate type"""
        if not value:
            return None
        
        print("===value===", value)
        param_type = param_info.get('type', 'string')
        
        try:
            if param_type == 'integer':
                # Extract first number found
                import re
                numbers = re.findall(r'-?\d+', value)
                return int(numbers[0]) if numbers else None
            elif param_type in ['number', 'float']:
                # Extract first decimal number found  
                import re
                numbers = re.findall(r'-?\d*\.?\d+', value)
                return float(numbers[0]) if numbers else None
            elif param_type == 'boolean':
                return value.lower() in ['true', '1', 'yes', 'on']
            else:
                # For string type, return clean value
                return value
        except:
            return None

class EnhancedToolParser(BaseOutputParser):
    """Enhanced tool parser with AI model support"""
    
    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, 'ai_extractor', AIToolExtractor())
    
    def parse(self, kb_context: str, text: str) -> ToolCall:
        if self.ai_extractor:
            tool_name, params, confidence = self.ai_extractor.extract_tool_info(text, MCP_TOOLS_REGISTRY)
            print("===tool_name===", tool_name)
            print("===params===", params)

            if "prompt" in params:
                params["prompt"] = params["prompt"] + "\n\n" + kb_context

            return ToolCall(tool_name=tool_name, parameters=params, confidence=confidence)
        
        # 4. Fallback to semantic matching
        return self._semantic_match_with_template(text)
    
    def _parse_natural_params(self, param_text: str) -> Dict[str, Any]:
        """Parse parameters from natural language"""
        params = {}
        if not param_text.strip():
            return params
        
        # Common patterns
        patterns = [
            (r'prompt:\s*([^,]+?)(?:,|$)', 'prompt'),
            (r'model\s+is\s+([^\s,]+)', 'model'),
            (r'temperature:\s*([0-9.]+)', 'temperature'),
            (r'max_tokens:\s*(\d+)', 'max_tokens'),
        ]
        
        for pattern, param_name in patterns:
            match = re.search(pattern, param_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Convert numeric values
                if param_name in ['temperature'] and value.replace('.', '').isdigit():
                    params[param_name] = float(value)
                elif param_name in ['max_tokens'] and value.isdigit():
                    params[param_name] = int(value)
                else:
                    params[param_name] = value
        
        # If no specific patterns found, treat as prompt
        if not params and param_text.strip():
            params['prompt'] = param_text.strip()
        
        return params
    
    def _semantic_match_with_template(self, text: str) -> ToolCall:
        """Fallback semantic matching"""
        if not MCP_TOOLS_REGISTRY:
            return ToolCall()
        
        best_tool = None
        best_score = 0
        
        text_lower = text.lower()
        for tool_name, info in MCP_TOOLS_REGISTRY.items():
            # Score by description similarity
            desc_words = info.get('description', '').lower().split()
            score = sum(1 for word in desc_words if word in text_lower)
            
            # Boost for name matches
            name_parts = tool_name.lower().split('.')
            if any(part in text_lower for part in name_parts):
                score += 2
            
            if score > best_score:
                best_score = score
                best_tool = tool_name
        
        if best_tool and best_score > 0:
            params = self._extract_params_from_text(best_tool, text)
            confidence = min(best_score * 0.2, 0.7)
            return ToolCall(tool_name=best_tool, parameters=params, confidence=confidence)
        
        return ToolCall()
    
    def _extract_params_from_text(self, tool_name: str, text: str) -> Dict[str, Any]:
        """Extract parameters using improved heuristics"""
        tool_info = MCP_TOOLS_REGISTRY.get(tool_name, {})
        params_schema = tool_info.get('parameters', {})
        
        extracted = {}
        for param_name, param_info in params_schema.items():
            if param_name in ['prompt', 'query', 'text', 'message']:
                # Clean main request
                clean_text = re.sub(r'^(please |can you |generate |create |using action [^\s]+ with param is )', '', text, flags=re.IGNORECASE).strip()
                extracted[param_name] = clean_text
                
            elif param_name == 'model' and 'model' in text.lower():
                model_match = re.search(r'model[:\s]+([a-zA-Z0-9-_]+)', text, re.IGNORECASE)
                if model_match:
                    extracted[param_name] = model_match.group(1)
        
        return extracted
             
class MCPToolsManager:
    """Enhanced MCP Tools Manager with PromptTemplate support"""
    TOOLS = []
    RESULTS = None
    
    @staticmethod
    async def discover_tools_from_sse(endpoint: str, auth_token: str = None):
        """Auto-discover tools from SSE endpoint using MCP client"""
        
        try:
            # Build args properly
            args = ["-y", "supergateway", "--sse", endpoint, "--allow-http", "--quiet"]
            if auth_token:
                args.extend(["--auth", auth_token])
            
            server_params = StdioServerParameters(
                command="npx",
                args=args
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                
                    MCPToolsManager.TOOLS = []
                    if hasattr(tools_response, 'tools') and tools_response.tools:
                        for tool in tools_response.tools:
                            try:
                                # Handle both dict and object formats
                                if hasattr(tool, 'dict'):
                                    tool_data = tool.dict()
                                elif hasattr(tool, '__dict__'):
                                    tool_data = tool.__dict__
                                else:
                                    tool_data = tool
                                
                                tool_info = {
                                    "name": tool_data.get("name", ""),
                                    "description": tool_data.get("description", ""),
                                    "inputSchema": tool_data.get("inputSchema", {}),
                                    "examples": []
                                }
                                
                                if tool_info["name"]:
                                    MCPToolsManager.TOOLS.append(tool_info)
                                    
                            except Exception:
                                continue
                    
                    # Connection sẽ đóng ở đây nhưng data đã lưu trong TOOLS
            # Return sau khi connection đã đóng an toàn
            return {"tools": MCPToolsManager.TOOLS, "count": len(MCPToolsManager.TOOLS)} if MCPToolsManager.TOOLS else {"error": "No tools discovered"}
                        
        except Exception as e:
            return {"error": f"Discovery failed: {str(e)}"}
          
    @staticmethod
    def register_remote_mcp_from_payload(payload: dict):
        """Enhanced registration with auto-discovery support"""
        name = payload.get("name")
        endpoint = payload.get("endpoint") 
        auth_token = payload.get("auth_token")
        tools = payload.get("tools", [])
        
        if not name or not endpoint:
            return {"error": "Missing required fields: name, endpoint"}
        
        # Auto-discover tools if not provided
        if not tools:
            try:
                import concurrent.futures
                
                def run_discovery():
                    # Create new event loop in separate thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            MCPToolsManager.discover_tools_from_sse(endpoint, auth_token)
                        )
                        # Đảm bảo copy data trước khi đóng loop
                        return dict(result) if isinstance(result, dict) else result
                    finally:
                        loop.close()
                
                # Run in thread pool to avoid event loop conflicts
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_discovery)
                    discovery_result = future.result(timeout=30)
                    
                if discovery_result and "error" not in discovery_result:
                    tools = discovery_result.get("tools", [])
                else:
                    return discovery_result or {"error": "Discovery failed"}
                
            except Exception as e:
                tools = MCPToolsManager.TOOLS
        
        # Store connection
        REMOTE_MCP_CONNECTIONS[name] = {
            "endpoint": endpoint,
            "auth_token": auth_token, 
            "tools": tools,
            "payload": payload
        }
        
        # Register tools with auto-generated examples
        registered_tools = []
        for tool in tools:
            tool_key = f"{name}.{tool['name']}"
            schema = tool.get('inputSchema', {})
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            # Auto-generate examples
            examples = MCPToolsManager._generate_examples(properties, required)
            
            MCP_TOOLS_REGISTRY[tool_key] = {
                "name": tool['name'],
                "description": tool.get('description', ''),
                "parameters": properties,
                "required": required,
                "examples": examples,
                "mcp_connection": name
            }
            registered_tools.append(tool_key)
        
        return {
            "message": f"Registered MCP '{name}' with {len(registered_tools)} tools" + 
                      (" (auto-discovered)" if tools else " (no tools discovered)"),
            "registered_tools": registered_tools,
            "endpoint": endpoint,
            "auto_discovered": len(tools) > 0
        }
    
    @staticmethod
    def _generate_examples(properties: dict, required: list) -> list:
        """Generate usage examples from schema"""
        if not properties or not required:
            return []
        
        examples = []
        
        # Basic example with required params
        basic_example = {}
        for param in required:
            if param in properties:
                param_info = properties[param]
                basic_example[param] = MCPToolsManager._get_example_value(param, param_info)
        
        if basic_example:
            examples.append(basic_example)
            
            # Full example with optional params
            full_example = basic_example.copy()
            for param, info in properties.items():
                if param not in required and 'default' in info:
                    full_example[param] = info['default']
            
            if full_example != basic_example:
                examples.append(full_example)
        
        return examples
    
    @staticmethod
    def _get_example_value(param_name: str, param_info: dict):
        """Generate example value for parameter"""
        if 'enum' in param_info:
            return param_info['enum'][0]
        if 'default' in param_info:
            return param_info['default']
        
        param_type = param_info.get('type', 'string')
        if param_type == 'string':
            return f"example_{param_name}" if param_name != 'prompt' else "Generate a story"
        elif param_type == 'integer':
            return 1
        elif param_type == 'boolean':
            return True
        elif param_type == 'number':
            return 1.0
        else:
            return f"<{param_name}>"
    
    @staticmethod
    def call_remote_tool_sse(tool_name: str, params: Dict[str, Any]):
        """Single call MCP tool - handle existing event loop"""
        if tool_name not in MCP_TOOLS_REGISTRY:
            return {"error": f"Tool '{tool_name}' not found"}
        
        # Validate parameters
        validation = MCPToolsManager.validate_parameters(tool_name, params)
        if not validation["valid"]:
            print("error", f"Validation failed: {validation['message']}")
            return False
        
        mcp_name = tool_name.split('.')[0]
        actual_tool_name = '.'.join(tool_name.split('.')[1:])
        connection = REMOTE_MCP_CONNECTIONS[mcp_name]
        
        try:
            import asyncio
            import concurrent.futures
            
            def run_in_thread():
                """Run MCP call in separate thread with new event loop"""
                # Tạo event loop mới trong thread riêng
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def single_call():
                    args = ["-y", "supergateway", "--sse", connection["endpoint"], "--allow-http"]
                    if connection.get("auth_token"):
                        args.extend(["--auth", connection["auth_token"]])
                    
                    server_params = StdioServerParameters(command="npx", args=args)
                    
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(actual_tool_name, params)
                            return result
                
                try:
                    # Chạy với timeout trong event loop mới
                    result = loop.run_until_complete(
                        asyncio.wait_for(single_call(), timeout=180.0)
                    )
                    return result
                except asyncio.TimeoutError:
                    return {"error": "Tool call timed out after 30 seconds"}
                except Exception as e:
                    return {"error": f"Tool execution failed: {str(e)}"}
                finally:
                    loop.close()
            
            # Chạy trong thread riêng để tránh conflict với existing loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                result = future.result(timeout=181.0)  # Timeout hơi dài hơn async timeout
                return result
                
        except concurrent.futures.TimeoutError:
            print("error", "Thread execution timed out")
            return False
        except Exception as e:
            print("error", f"Setup failed: {str(e)}")
            return False
    
    @staticmethod
    def validate_parameters(tool_name: str, params: Dict[str, Any]):
        """Quick parameter validation"""
        if tool_name not in MCP_TOOLS_REGISTRY:
            return {"valid": False, "message": f"Tool '{tool_name}' not found"}
        
        tool_info = MCP_TOOLS_REGISTRY[tool_name]
        required_params = tool_info.get('required', [])
        
        # Check required parameters
        missing = [p for p in required_params if p not in params]
        if missing:
            return {"valid": False, "message": f"Missing required: {', '.join(missing)}"}
        
        return {"valid": True, "message": "Valid"}
    
    @staticmethod
    def fast_extract_tool_enhanced(kb_context: str, prompt: str):
        """Enhanced tool extraction using PromptTemplate"""
        if not MCP_TOOLS_REGISTRY:
            return None, prompt, "no_tools"
        
        # Use enhanced parser with template support
        parser = EnhancedToolParser()
        
        try:
            result = parser.parse(kb_context, prompt)
            if result.tool_name and result.tool_name in MCP_TOOLS_REGISTRY:
                return result.tool_name, result.parameters, f"enhanced_{result.confidence:.2f}"
        except Exception as e:
            logger.debug(f"Enhanced parsing failed: {e}")
        
        return None, prompt, "no_match"

class MyModel(AIxBlockMLBase):
    """Main model class with enhanced MCP tools management"""
    
    @mcp.tool()
    def action(self, command, **kwargs):
        logger.info(f"Received command: {command} with args: {kwargs}")
        
        # 🔧 MCP Tools Management Commands
        if command.lower() == "mcp_register_payload":
            result = MCPToolsManager.register_remote_mcp_from_payload(kwargs)
            
            # Handle knowledge base connections
            name = kwargs.get("name", "")
            memory_conn = kwargs.get("memoryConnection")
            storage_conn = kwargs.get("storageConnection")
            
            if name and (memory_conn or storage_conn):
                kb_result = KnowledgeBaseManager.register_connections(name, memory_conn, storage_conn)
                result["knowledge_base"] = kb_result

                update_result = KnowledgeBaseManager.update_knowledge_base(name)
                result["kb_update"] = update_result
            
            return result
        
        elif command.lower() == "mcp_call":
            tool_name = kwargs.get("tool_name")
            params = kwargs.get("params", {})
            return MCPToolsManager.call_remote_tool_sse(tool_name, params)
        
        elif command.lower() == "mcp_list_tools":
            return {
                "tools": {name: info for name, info in MCP_TOOLS_REGISTRY.items()},
                "count": len(MCP_TOOLS_REGISTRY)
            }
        
        elif command.lower() == "kb_search":
            name = kwargs.get("name", "")
            query = kwargs.get("query", "")
            limit = kwargs.get("limit", 5)
            return {"result": KnowledgeBaseManager.search_knowledge(name, query, limit)}
        
        elif command.lower() == "kb_update":
            name = kwargs.get("name", "")
            return KnowledgeBaseManager.update_knowledge_base(name)
        
        # 📝 Prompt Template Management Commands
        elif command.lower() == "template_update":
            template_name = kwargs.get("template_name", "")
            template_text = kwargs.get("template_text", "")
            input_variables = kwargs.get("input_variables", [])
            description = kwargs.get("description", "")
            
            if not template_name or not template_text:
                return {"error": "template_name and template_text required"}
            
            # Validate first
            validation = PromptTemplateManager.validate_template(template_text, input_variables)
            if not validation["valid"]:
                return {"error": f"Template validation failed: {validation['message']}"}
            
            return PromptTemplateManager.update_template(template_name, template_text, input_variables, description)
        
        elif command.lower() == "template_list":
            return PromptTemplateManager.list_templates()
        
        elif command.lower() == "template_get":
            template_name = kwargs.get("template_name", "")
            if not template_name:
                return {"error": "template_name required"}
            
            if template_name not in CUSTOM_TEMPLATES:
                return {"error": f"Template '{template_name}' not found"}
            
            return {
                "template": CUSTOM_TEMPLATES[template_name],
                "name": template_name
            }
        
        elif command.lower() == "template_delete":
            template_name = kwargs.get("template_name", "")
            if not template_name:
                return {"error": "template_name required"}
            
            return PromptTemplateManager.delete_template(template_name)
        
        elif command.lower() == "template_validate":
            template_text = kwargs.get("template_text", "")
            input_variables = kwargs.get("input_variables", [])
            
            if not template_text:
                return {"error": "template_text required"}
            
            return PromptTemplateManager.validate_template(template_text, input_variables)

        elif command.lower() == "predict":
            # 🤖 Enhanced Agent Predict Command with PromptTemplate
            
            # Extract parameters
            prompt = kwargs.get("prompt") or kwargs.get("text", "")
            model_id = kwargs.get("model_id", "Qwen/Qwen3-1.7B")
            session_id = kwargs.get("session_id")
            use_history = kwargs.get("use_history", True)
            enable_function_calling = kwargs.get("enable_function_calling", True)
            use_knowledge_base = kwargs.get("use_knowledge_base", True)
            max_history = kwargs.get("max_history", 5)
            hf_token = kwargs.get("push_to_hub_token", "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")
            template_name = kwargs.get("template_name", "context_assembly")
            
            if not prompt:
                return {"error": "Prompt required", "session_id": session_id}
            
            # 📚 Knowledge Base Context
            kb_context = ""
            kb_used = []
            if use_knowledge_base and KNOWLEDGE_BASES:
                kb_context = KnowledgeBaseManager.search_all_knowledge_bases(prompt, limit=10)
                if kb_context:
                    kb_used = list(KNOWLEDGE_BASES.keys())
                    logger.info(f"📚 KB context from {len(kb_used)} bases: {len(kb_context)} chars")
            
            # 📝 Session Management
            from utils.chat_history import ChatHistoryManager
            chat_manager = ChatHistoryManager(persist_directory="./chroma_db_history")
            
            if not session_id:
                session = chat_manager.create_new_session()
                session_id = session["session_id"]
                logger.info(f"🆕 Session created: {session_id}")
            
            conversation_history = []
            if use_history:
                conversation_history = chat_manager.get_session_history(session_id, limit=max_history)
                logger.info(f"📚 Loaded {len(conversation_history)} messages")
            
            # ⚡ Enhanced Tool Detection with PromptTemplate
            function_calls = []
            final_response = ""
            thinking = ""
            
            if enable_function_calling and MCP_TOOLS_REGISTRY:
                # Enhanced tool extraction using templates
                
                tool_name, tool_params, extraction_method = MCPToolsManager.fast_extract_tool_enhanced(kb_context, prompt)
                
                if tool_name:
                    logger.info(f"🔧 Tool detected: {tool_name} via {extraction_method}")
                    
                    # Add KB context to tool parameters if relevant
                    if kb_context and 'context' not in tool_params and 'knowledge' not in tool_params:
                        # Check if tool accepts context parameter
                        tool_info = MCP_TOOLS_REGISTRY.get(tool_name, {})
                        tool_schema = tool_info.get('parameters', {})
                        
                        # Add context if tool has context/knowledge parameters
                        for param_name in tool_schema.keys():
                            if any(keyword in param_name.lower() for keyword in ['context', 'knowledge', 'background', 'info']):
                                tool_params[param_name] = kb_context[:1000]  # Limit context size
                                logger.info(f"📚 Added KB context to {param_name}")
                                break
                    
                    # Execute tool directly
                    result = MCPToolsManager.call_remote_tool_sse(tool_name, tool_params)
                    print("result", result)
                    if result and result != False:
                        print("Process result")
                        # Process result
                        if hasattr(result, 'content'):
                            if hasattr(result.content, '__iter__') and not isinstance(result.content, str):
                                content_text = ""
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        content_text += item.text
                                    elif isinstance(item, dict) and 'text' in item:
                                        content_text += item['text']
                                    else:
                                        content_text += str(item)
                                final_response = content_text
                            else:
                                final_response = str(result.content)
                        elif isinstance(result, dict):
                            if "error" in result:
                                final_response = ""
                            else:
                                content = result.get('content', str(result))
                                # Extract meaningful content from JSON
                                if '"output":' in content:
                                    try:
                                        import re
                                        output_match = re.search(r'"output":\s*"([^"]*(?:\\.[^"]*)*)"', content)
                                        if output_match:
                                            content = output_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                    except:
                                        pass
                                final_response = content
                        else:
                            final_response = str(result)
                        
                        # Record function call
                        function_calls = [{
                            "tool": tool_name,
                            "params": tool_params,
                            "result": {"content": final_response, "success": "error" not in str(result).lower()},
                            "success": True,
                            "extraction_method": extraction_method
                        }]
                        
                        thinking = f"Enhanced tool extraction: {tool_name} via {extraction_method}. KB context: {len(kb_context)} chars"
            # 🤖 Fallback to model generation if no tool detected
            if not final_response:
                logger.info("🤖 No tool detected or tool failed, using model generation")
                
                # 🏭 Load Model
                def ensure_model_loaded():
                    global pipe_prediction, tokenizer, model_predict
                    
                    if pipe_prediction and model_predict == model_id:
                        return
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Determine source
                    model_name = model_id.split("/")[-1]
                    local_path = f"./data/checkpoint/{model_name}"
                    source = local_path if os.path.exists(f"{local_path}/config.json") else model_id
                    
                    if hf_token:
                        login(token=hf_token)
                    
                    tokenizer = AutoTokenizer.from_pretrained(source)
                    
                    if torch.cuda.is_available():
                        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        pipe_prediction = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, device_map="auto")
                    else:
                        pipe_prediction = AutoModelForCausalLM.from_pretrained(source, device_map="cpu")
                    
                    model_predict = model_id
                    logger.info(f"✅ Model loaded: {model_id}")
                
                ensure_model_loaded()
                
                with torch.no_grad():
                    # 🗨️ Build Messages using PromptTemplate
                    messages = []
                    
                    # Enhanced system prompt using template
                    if kb_context or MCP_TOOLS_REGISTRY:
                        system_template = PromptTemplateManager.get_context_assembly_template(template_name)
                        history_text = "\n".join([f"User: {turn.get('user_message', '')}\nBot: {turn.get('bot_response', '')}" 
                                            for turn in conversation_history[-10:]])
                        system_msg = system_template.format(
                            history=history_text,
                            kb_context=kb_context
                        )
                        messages.append({"role": "system", "content": system_msg})
                    
                    # # Add history
                    # for turn in conversation_history:
                    #     user_msg = turn.get('user_message', '').strip()
                    #     bot_msg = turn.get('bot_response', '').strip()
                    #     if user_msg and bot_msg:
                    #         messages.extend([
                    #             {"role": "user", "content": user_msg},
                    #             {"role": "assistant", "content": bot_msg}
                    #         ])
                    
                    # Current prompt
                    messages.append({"role": "user", "content": prompt})

                    print("===messages===", messages)
                    
                    # 🚀 Generate
                    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([chat_text], return_tensors="pt").to(pipe_prediction.device)
                    
                    outputs = pipe_prediction.generate(
                        **inputs,
                        max_new_tokens=32768
                    )
                    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                    try:
                        # rindex finding 151668 (</think>)
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0
                        
                    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    final_response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                    function_calls = []
            
            # 💾 Save to History
            predictions = [
                {
                    "result": [
                        {
                            "from_name": "generated_text",
                            "to_name": "text_output", 
                            "type": "textarea",
                            "value": {
                                "thinking": [thinking], 
                                "text": [final_response]
                            },
                        }
                    ],
                    "model_version": model_id,
                }
            ]

            # 💾 Save to History
            if use_history:
                try:
                    chat_manager.save_conversation_turn(
                        session_id=session_id,
                        user_message=prompt,
                        bot_response=final_response,
                        metadata={
                            "model_id": model_id,
                            "function_calls": len(function_calls),
                            "thinking": thinking[:100] + "..." if len(thinking) > 100 else thinking,
                            "kb_used": kb_used,
                            "extraction_method": function_calls[0]["extraction_method"] if function_calls else "none",
                            "prompt_template_used": True
                        }
                    )
                    logger.info(f"💾 Saved to session {session_id}")
                except Exception as e:
                    logger.error(f"❌ Save failed: {e}")

            # 🎯 Return Response with new format
            return {
                "success": True,
                "message": "predict completed successfully",
                "result": predictions,
                "text": final_response,
                "session_id": session_id,
                "metadata": {
                    "model_id": model_id,
                    "history_turns": len(conversation_history),
                    "tools_used": len(function_calls),
                    "kb_bases_used": kb_used,
                    "extraction_method": function_calls[0]["extraction_method"] if function_calls else "model_generation",
                    "kb_context_chars": len(kb_context),
                    "prompt_template_enhanced": True
                }
            }

    @mcp.tool()
    def model(self, **kwargs):
        global model_demo, tokenizer_demo, model_loaded_demo, model_id_demo

        model_id_demo = kwargs.get("model_id", "Qwen/Qwen3-1.7B")
        project_id = kwargs.get("project_id", 0)

        print(
            f"""\
        Project ID: {project_id}
        """
        )

        hf_access_token = kwargs.get(
            "hf_access_token", "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY"
        )
        # login(token=hf_access_token)
        MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

        DESCRIPTION = """\
        # Qwen/Qwen3
        """

        if not torch.cuda.is_available():
            DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        def load_model(model_id, temperature, top_p, top_k, max_new_token):
            print(
                f"""\
                temperature: {temperature}
                top_p: {top_p}
                top_k: {top_k}
                max_new_token: {max_new_token}
                """
            )
            global model_demo, tokenizer_demo, model_loaded_demo

            if torch.cuda.is_available() and not model_loaded_demo:
                model_demo = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    token=hf_access_token,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                )
                tokenizer_demo = AutoTokenizer.from_pretrained(
                    model_id, token=hf_access_token
                )
                tokenizer_demo.use_default_system_prompt = False
                model_loaded_demo = True
                return f"Model {model_id} loaded successfully!"
            elif model_loaded_demo:
                return "Model is already loaded! Please refresh the page to load a different model."
            else:
                return "Error: CUDA is not available!"

        @spaces.GPU
        def generate(
            message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1,
        ) -> Iterator[str]:
            if not model_loaded_demo:
                return (
                    "Please load the model first by clicking the 'Load Model' button."
                )
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": str(system_prompt)})

            # Add chat history
            for user_msg, assistant_msg in chat_history:
                chat_messages.append({"role": "user", "content": str(user_msg)})
                chat_messages.append(
                    {"role": "assistant", "content": str(assistant_msg)}
                )

            # Add the current message
            chat_messages.append({"role": "user", "content": str(message)})
            text = tokenizer_demo.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer_demo([text], return_tensors="pt").to(
                model_demo.device
            )
            if model_inputs.input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
                model_inputs.input_ids = model_inputs.input_ids[
                    :, -MAX_INPUT_TOKEN_LENGTH:
                ]
                gr.Warning(
                    f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens."
                )

            generated_ids = model_demo.generate(**model_inputs, max_new_tokens=512)

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer_demo.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return response

        chat_interface = gr.ChatInterface(
            fn=generate,
            stop_btn=gr.Button("Stop"),
            examples=[
                ["implement snake game using pygame"],
                [
                    "Can you explain briefly to me what is the Python programming language?"
                ],
                ["write a program to find the factorial of a number"],
            ],
        )

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=1):
                    load_btn = gr.Button("Load Model")
                with gr.Column(scale=1):
                    status_text = gr.Textbox(label="Model Status", interactive=False)

            with gr.Accordion("Advanced Options", open=False):
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=100.0, step=0.1, value=0.9
                )
                top_p = gr.Slider(
                    label="Top_p", minimum=0.0, maximum=1.0, step=0.1, value=0.6
                )
                top_k = gr.Slider(
                    label="Top_k", minimum=0, maximum=100, step=1, value=0
                )
                max_new_token = gr.Slider(
                    label="Max new tokens", minimum=1, maximum=1024, step=1, value=256
                )
            load_btn.click(fn=lambda: load_model(model_id_demo, temperature.value, top_p, top_k, max_new_token), outputs=status_text)
    
            chat_interface.render()

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )
        return {"share_url": share_url, "local_url": local_url}
    
    @mcp.tool()
    def model_trial(self, project, **kwargs):
        return {"message": "Done", "result": "Done"}

    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)