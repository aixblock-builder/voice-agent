from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from langchain.schema import BaseOutputParser
import re
import asyncio
from loguru import logger

# 🔧 MCP Tools Registry
MCP_TOOLS_REGISTRY = {}
TOOLS = []
REMOTE_MCP_CONNECTIONS = {}

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