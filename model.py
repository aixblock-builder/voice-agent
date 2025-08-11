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
import redis
import boto3
import ftplib
import paramiko
from smb.SMBConnection import SMBConnection
from typing import List, Dict, Any
import hashlib
import io
import tempfile

# ------------------------------------------------------------------------------
HfFolder.save_token("hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")
login(token = "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")

CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
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
        
        for obj in response.get('Contents', []):
            if obj['Key'].endswith(('.txt', '.md', '.json', '.pdf', '.docx')):
                try:
                    file_obj = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
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
                    kb_context = f"[From {kb_name}]:\n" + "\n".join([doc.page_content for doc in docs])
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
        
class MCPToolsManager:
    """Simplified MCP Tools Manager with auto-discovery support"""
    TOOLS = []  # Biến lưu trữ tools tạm thời
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
            return {"error": f"Validation failed: {validation['message']}"}
        
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
                        asyncio.wait_for(single_call(), timeout=60.0)
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
                result = future.result(timeout=35.0)  # Timeout hơi dài hơn async timeout
                return result
                
        except concurrent.futures.TimeoutError:
            return {"error": "Thread execution timed out"}
        except Exception as e:
            return {"error": f"Setup failed: {str(e)}"}
    
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
    def generate_tools_prompt():
        """Generate concise tools description for AI"""
        if not MCP_TOOLS_REGISTRY:
            return "No tools available."
        
        prompt = "Available tools:\n\n"
        for tool_name, info in MCP_TOOLS_REGISTRY.items():
            prompt += f"🔧 **{tool_name}**\n"
            prompt += f"   {info['description']}\n"
            
            # Show key parameters
            params = info.get('parameters', {})
            if params:
                key_params = []
                for param, details in params.items():
                    param_type = details.get('type', 'string')
                    if details.get('default'):
                        key_params.append(f"{param}({param_type}, default: {details['default']})")
                    else:
                        key_params.append(f"{param}({param_type})")
                prompt += f"   Parameters: {', '.join(key_params[:5])}{'...' if len(key_params) > 5 else ''}\n"
            
            if info['required']:
                prompt += f"   Required: {', '.join(info['required'])}\n"
            
            if info['examples']:
                example = info['examples'][0]
                prompt += f"   Example: TOOL_CALL:{tool_name}:{json.dumps(example)}\n"
            
            prompt += "\n"
        
        return prompt

class MyModel(AIxBlockMLBase):
    """Main model class with MCP tools management"""
    
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

        elif command.lower() == "predict":
            # 🤖 Streamlined Agent Predict Command
            
            # Extract parameters
            prompt = kwargs.get("prompt") or kwargs.get("text", "")
            model_id = kwargs.get("model_id", "Qwen/Qwen3-1.7B")
            session_id = kwargs.get("session_id")
            use_history = kwargs.get("use_history", True)
            enable_function_calling = kwargs.get("enable_function_calling", True)
            use_knowledge_base = kwargs.get("use_knowledge_base", True)
            max_history = kwargs.get("max_history", 5)
            hf_token = kwargs.get("push_to_hub_token", "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")
            
            if not prompt:
                return {"error": "Prompt required", "session_id": session_id}
            
            kb_context = ""
            kb_used = []
            if use_knowledge_base and KNOWLEDGE_BASES:
                kb_context = KnowledgeBaseManager.search_all_knowledge_bases(prompt, limit=3)
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
                # 🗨️ Build Messages
                messages = []
                
                # System prompt for function calling
                if enable_function_calling and MCP_TOOLS_REGISTRY:
                    tools_info = MCPToolsManager.generate_tools_prompt()
                    messages.append({
                        "role": "system", 
                        "content": f"""You are an intelligent assistant with access to tools:

        {tools_info}

        FUNCTION CALLING PARSING RULES:
        1. **Split user request into MAIN TASK + PARAMETERS**
        2. **Main task**: What user wants (before "using", "with", "set")  
        3. **Parameters**: Specific settings mentioned after "using", "with", "set"
        4. **Format**: TOOL_CALL:tool_name:{{"prompt":"main_task","param":"value"}}

        PARSING PATTERNS:
        - "Generate X using model Y" → prompt="Generate X", model="Y"
        - "Create X, using model in params is Y" → prompt="Create X", model="Y" 
        - "Write X with temperature 0.5" → prompt="Write X", temperature=0.5
        - "Generate X, set model to Y and temperature to Z" → prompt="Generate X", model="Y", temperature=Z

        EXAMPLES:
        User: "Generate a short story using gpt-4-turbo"
        → TOOL_CALL:mcp.openai-ask_chatgpt:{{"prompt": "Generate a short story", "model": "gpt-4-turbo"}}

        User: "Write a poem with temperature 0.8 and model gpt-4"
        → TOOL_CALL:mcp.openai-ask_chatgpt:{{"prompt": "Write a poem", "model": "gpt-4", "temperature": 0.8}}

        **KEY**: Extract the MAIN REQUEST (what to generate) as prompt, then add mentioned parameters."""
                    })
                
                if kb_context:
                    messages.append({
                        "role": "system",
                        "content": f"""KNOWLEDGE BASE CONTEXT:
        {kb_context}

        Use this context to answer questions when relevant. Always prioritize accuracy from the knowledge base over general knowledge."""
                    })
                
                # Add history
                for turn in conversation_history:
                    user_msg = turn.get('user_message', '').strip()
                    bot_msg = turn.get('bot_response', '').strip()
                    if user_msg and bot_msg:
                        messages.extend([
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": bot_msg}
                        ])
                
                # Current prompt
                messages.append({"role": "user", "content": prompt})
                
                # 🚀 Generate
                chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([chat_text], return_tensors="pt").to(pipe_prediction.device)
                
                outputs = pipe_prediction.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Extract response
                output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                try:
                    thinking_end = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    thinking_end = 0
                
                thinking = tokenizer.decode(output_ids[:thinking_end], skip_special_tokens=True).strip()
                response = tokenizer.decode(output_ids[thinking_end:], skip_special_tokens=True).strip()
                
                # 🔧 Process Function Calls
                function_calls = []
                final_response = response
                
                if enable_function_calling and "TOOL_CALL:" in response:
                    lines = []
                    for line in response.split('\n'):
                        line = line.strip()
                        if line.startswith("TOOL_CALL:"):
                            try:
                                # Parse: TOOL_CALL:tool_name:{"param": "value"}
                                parts = line[10:].split(':', 1)
                                if len(parts) != 2:
                                    lines.append("❌ Invalid tool format")
                                    continue
                                
                                tool_name, params_str = parts
                                params = json.loads(params_str.strip()) if params_str.strip() else {}
                                
                                # Execute tool
                                result = MCPToolsManager.call_remote_tool_sse(tool_name.strip(), params)
                                
                                # Handle CallToolResult object properly
                                if hasattr(result, 'content'):
                                    # Extract content from MCP CallToolResult
                                    if hasattr(result.content, '__iter__') and not isinstance(result.content, str):
                                        # Handle list of content items
                                        content_text = ""
                                        for item in result.content:
                                            if hasattr(item, 'text'):
                                                content_text += item.text
                                            elif isinstance(item, dict) and 'text' in item:
                                                content_text += item['text']
                                            else:
                                                content_text += str(item)
                                        result_content = content_text
                                    else:
                                        result_content = str(result.content)
                                    
                                    processed_result = {
                                        "content": result_content,
                                        "success": True
                                    }
                                elif isinstance(result, dict):
                                    processed_result = result
                                else:
                                    processed_result = {"content": str(result), "success": True}
                                
                                function_calls.append({
                                    "tool": tool_name.strip(),
                                    "params": params,
                                    "result": processed_result,
                                    "success": processed_result.get("success", "error" not in processed_result)
                                })
                                
                                # Add result to response
                                if "error" in processed_result:
                                    lines.append(f"❌ {tool_name}: {processed_result['error']}")
                                else:
                                    content = processed_result.get('content', str(processed_result))
                                    # Extract meaningful content, skip technical info
                                    if isinstance(content, str):
                                        # Try to extract the actual response from JSON or technical output
                                        if '"output":' in content:
                                            try:
                                                import re
                                                output_match = re.search(r'"output":\s*"([^"]*(?:\\.[^"]*)*)"', content)
                                                if output_match:
                                                    content = output_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                            except:
                                                pass
                                        
                                        # Truncate for display
                                        display_content = content[:200] + "..." if len(content) > 200 else content
                                        lines.append(f"✅ {tool_name} result:\n{display_content}")
                                    else:
                                        lines.append(f"✅ {tool_name}: Executed successfully")
                                    
                            except Exception as e:
                                lines.append(f"❌ Tool error: {str(e)}")
                                logger.error(f"Function call failed: {e}")
                        else:
                            lines.append(line)
                    
                    final_response = '\n'.join(lines)
                
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
                                "thinking": thinking[:100] + "..." if len(thinking) > 100 else thinking
                            }
                        )
                        logger.info(f"💾 Saved to session {session_id}")
                    except Exception as e:
                        logger.error(f"❌ Save failed: {e}")
                
                # 🎯 Return Response
                return {
                    "success": True,
                    "session_id": session_id,
                    "response": final_response,
                    "thinking": thinking,
                    "function_calls": function_calls,
                    "metadata": {
                        "model_id": model_id,
                        "history_turns": len(conversation_history),
                        "tools_used": len(function_calls)
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