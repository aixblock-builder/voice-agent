from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
import hashlib
from loguru import logger
import redis
import ftplib
from typing import List
import io
import paramiko
import boto3
from smb.SMBConnection import SMBConnection
import tempfile

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