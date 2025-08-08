import asyncio
import json
import os
import re
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
from model_docchat import docchat_answer

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
                import threading
                
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
            import threading
            
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

        elif command.lower() == "predict":
            prompt = kwargs.get("prompt", None)
            model_id = kwargs.get("model_id", "Qwen/Qwen3-1.7B")
            text = kwargs.get("text", None)
            task = kwargs.get("task", "")
            raw_input = kwargs.get("input", None)
            docchat_mode = kwargs.get("docchat", False)
            doc_files = kwargs.get("doc_files", None)
            conversation_history = kwargs.get("conversation_history", [])
            session_id = kwargs.get("session_id", None)
            use_history = kwargs.get("use_history", True)
            hf_access_token = kwargs.get("push_to_hub_token", "hf_"+"bjIxyaTXDGqlUa"+"HjvuhcpfVkPjcvjitRsY")
            # 🔧 Function calling support
            enable_function_calling = kwargs.get("enable_function_calling", True)
            available_tools = kwargs.get("available_tools", [])

            # 🧠 CHAT HISTORY MANAGEMENT
            from utils.chat_history import ChatHistoryManager
            chat_history = ChatHistoryManager(persist_directory="./chroma_db_history")
            
            # Auto-create session if not provided
            if not session_id:
                session_result = chat_history.create_new_session()
                session_id = session_result["session_id"]
                print(f"🆕 Created new session: {session_id} with title: {session_result['title']}")
            
            # Load conversation history if enabled
            if use_history and not conversation_history:
                conversation_history = chat_history.get_session_history(session_id, limit=10)
                if conversation_history:
                    print(f"📚 Loaded {len(conversation_history)} previous conversations for session {session_id}")
                else:
                    print(f"📝 Starting new conversation for session {session_id}")
            
            original_prompt = prompt or text
            if hf_access_token:
                login(token=hf_access_token)
            
            if raw_input:
                input_datas = json.loads(raw_input)
                print(input_datas)

            predictions = []

            if not prompt or prompt == "":
                prompt = text

            # Check if any recent conversation history has doc_files
            history_has_docs = False
            history_doc_files = []
            if conversation_history:
                for turn in reversed(conversation_history):  
                    turn_doc_files = turn.get('doc_files', [])
                    if turn_doc_files:
                        latest_file = turn_doc_files[-1]
                        if latest_file:
                            history_has_docs = True
                            history_doc_files = [latest_file]
                            print(f"📄 Found latest doc_file in conversation history: {latest_file}")
                            break

            def smart_pipeline(
                model_id: str,
                token: str,
                local_dir="./data/checkpoint",
                task="text-generation",
            ):
                global pipe_prediction, tokenizer, model_predict
                model_predict = model_id

                print(f"model_predict: {model_predict}")

                if pipe_prediction == None:
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        model_name = model_id.split("/")[-1]
                        local_model_dir = os.path.join(local_dir, model_name)
                        if os.path.exists(local_model_dir) and os.path.exists(
                            os.path.join(local_model_dir, "config.json")
                        ):
                            print(f"✅ Loading model from local: {local_model_dir}")
                            model_source = local_model_dir
                        else:
                            print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                            model_source = model_id
                    except:
                        print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                        model_source = model_id

                    tokenizer = AutoTokenizer.from_pretrained(model_source)
                    if torch.cuda.is_available():
                        if torch.cuda.is_bf16_supported():
                            dtype = torch.bfloat16
                        else:
                            dtype = torch.float16

                        print("Using CUDA.")
                        pipe_prediction = AutoModelForCausalLM.from_pretrained(
                            model_source,
                            torch_dtype=dtype,
                            device_map="auto"
                        )
                    else:
                        print("Using CPU.")
                        pipe_prediction = AutoModelForCausalLM.from_pretrained(
                            model_source,
                            device_map="cpu",
                        )

            print(pipe_prediction, model_id)
            with torch.no_grad():
                # Load the model
                if not pipe_prediction or model_predict != model_id:
                    smart_pipeline(model_id, hf_access_token)

                # --- DOCCHAT INTEGRATION ---
                if docchat_mode or doc_files or history_has_docs:
                    if not doc_files:
                        doc_files = []
                    if isinstance(doc_files, str):
                        doc_files = [f.strip() for f in doc_files.split(",") if f.strip()]
                    
                    if not doc_files and history_has_docs:
                        doc_files = history_doc_files
                        print(f"🔄 Using doc_files from conversation history: {doc_files}")
                    
                    enhanced_prompt = prompt
                    if conversation_history and not docchat_mode:
                        history_context = chat_history.format_history_for_context(conversation_history, max_turns=3)
                        enhanced_prompt = f"{history_context}\n\nCurrent Question: {prompt}"
                        print(f"🔄 Using conversation history for session {session_id}")
                    
                    answer, verification = docchat_answer(enhanced_prompt, doc_files, model_id, pipe_prediction, tokenizer)
                    if verification != "" or docchat_mode:
                        predictions.append({
                            "result": [
                                {
                                    "from_name": "generated_text",
                                    "to_name": "text_output",
                                    "type": "textarea",
                                    "value": {
                                        "text": [answer],
                                        "thinking": [verification]
                                    },
                                }
                            ],
                            "model_version": "docchat"
                        })
                        
                        if use_history and original_prompt and answer:
                            try:
                                if history_has_docs and not docchat_mode and not kwargs.get("doc_files"):
                                    mode = "docchat_from_history"
                                else:
                                    mode = "docchat"
                                
                                chat_history.save_conversation_turn(
                                    session_id=session_id,
                                    user_message=original_prompt,
                                    bot_response=answer,
                                    doc_files=doc_files,
                                    metadata={"command": "predict", "mode": mode, "model_id": model_id, "history_docs_used": history_has_docs}
                                )
                                print(f"💾 Saved DocChat conversation to session {session_id} (mode: {mode})")
                            except Exception as e:
                                print(f"❌ Failed to save DocChat conversation: {e}")
                        
                        return {"message": "predict completed successfully (docchat)", "result": predictions, "session_id": session_id}

                # Prepare messages with conversation history
                messages = []
                
                if conversation_history:
                    print(f"🔄 Adding conversation history to messages for session {session_id}")
                    for turn in conversation_history[-3:]:
                        user_msg = turn.get('user_message', '')
                        bot_response = turn.get('bot_response', '')
                        if user_msg and bot_response:
                            messages.append({"role": "user", "content": user_msg})
                            messages.append({"role": "assistant", "content": bot_response})

                # 🔧 Enhanced tool system prompt with detailed schemas
                if enable_function_calling and (available_tools or MCP_TOOLS_REGISTRY):
                    # Generate detailed tools description
                    tools_prompt = MCPToolsManager.generate_tools_prompt()
                    
                    tools_info = f"""You are an intelligent assistant with access to tools. Available tools:

                {tools_prompt}

                FUNCTION CALLING RULES:
                1. **Extract ALL mentioned parameters** from user's request
                2. **Use EXACT parameter names** from schema
                3. **Include required + user-specified parameters**
                4. Format: TOOL_CALL:tool_name:{{"param1": "value1", "param2": "value2"}}

                PARAMETER EXTRACTION EXAMPLES:
                - User: "Generate a story using gpt-4-turbo" 
                → TOOL_CALL:mcp.openai-ask_chatgpt:{{"prompt": "Generate a story", "model": "gpt-4-turbo"}}
                - User: "Create image of cat with high quality"
                → TOOL_CALL:mcp.openai-generate_image:{{"prompt": "cat", "quality": "hd"}}
                - User: "Ask GPT with temperature 0.5: What is AI?"
                → TOOL_CALL:mcp.openai-ask_chatgpt:{{"prompt": "What is AI?", "temperature": 0.5}}

                **KEY: Extract parameters mentioned in user's request, don't add extras.**"""
                    
                    if messages:
                        messages.insert(0, {"role": "system", "content": tools_info})
                    else:
                        messages.append({"role": "system", "content": tools_info})
                
                messages.append({"role": "user", "content": prompt})
                print(messages)
                
                # 🔧 Multi-turn generation for better function calling responses
                function_call_results = []
                iteration = 0
            
                # Generate response
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(pipe_prediction.device)

                generated_ids = pipe_prediction.generate(
                    **model_inputs,
                    max_new_tokens=512,  # Increased for better responses
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                generated_text = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                print(f"Iteration {iteration}")
                print(f"thinking_content: {thinking_content}")
                print(f"generated_text: {generated_text}")

                # Check if there are function calls to process
                print(enable_function_calling, "TOOL_CALL:" in generated_text)
                if enable_function_calling and "TOOL_CALL:" in generated_text:
                    lines = generated_text.split('\n')
                    print(lines)
                    processed_lines = []
                    print("=====", len(lines))
                    for line in lines:
                        line = re.sub(r'^\s+', '', line)
                        print("line", line)
                        print("line.strip()", line.strip())
                        if line.strip().startswith("TOOL_CALL:"):
                            try:
                                # Remove "TOOL_CALL:" prefix
                                call_content = line[10:]
                                
                                # Find the first colon to separate tool_name and params
                                colon_pos = call_content.find(':')
                                if colon_pos == -1:
                                    processed_lines.append(f"❌ Tool call error: Invalid format")
                                    continue
                                    
                                tool_name = call_content[:colon_pos].strip()
                                params_str = call_content[colon_pos + 1:].strip()
                                
                                # Parse JSON params
                                if not params_str:
                                    params = {}
                                else:
                                    try:
                                        params = json.loads(params_str)
                                    except json.JSONDecodeError as je:
                                        processed_lines.append(f"❌ JSON parsing error: {str(je)}")
                                        continue
                                
                                print(f"🔧 Calling tool '{tool_name}' with params: {params}")
                                
                                # Call MCP tool với SSE
                                tool_result = MCPToolsManager.call_remote_tool_sse(tool_name, params)
                                print("==tool_result==", tool_result)
                                function_call_results.append({
                                    "tool": tool_name,
                                    "params": params,
                                    "result": tool_result
                                })
                                
                            except Exception as e:
                                processed_lines.append(f"❌ Tool execution error: {str(e)}")
                                print(f"Debug: Tool call exception: {str(e)} for line: {line}")
                        else:
                            processed_lines.append(line)
                    
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": "generated_text",
                            "to_name": "text_output",
                            "type": "textarea",
                            "value": {
                                "thinking": [thinking_content], 
                                "text": [generated_text],
                                "function_calls": function_call_results
                            },
                        }
                    ],
                    "model_version": f"v2.0-{model_id}",
                }
            )

            # 💾 Save conversation to history (Normal mode)
            if use_history and original_prompt and generated_text:
                try:
                    chat_history.save_conversation_turn(
                        session_id=session_id,
                        user_message=original_prompt,
                        bot_response=generated_text,
                        doc_files=[],
                        metadata={
                            "command": "predict", 
                            "mode": "enhanced_sse_function_calling" if function_call_results else "normal", 
                            "model_id": model_id, 
                            "thinking": thinking_content,
                            "function_calls": function_call_results,
                            "iterations": iteration
                        }
                    )
                    print(f"💾 Saved conversation to session {session_id}")
                except Exception as e:
                    print(f"❌ Failed to save conversation: {e}")

            return {"message": "predict completed successfully", "result": predictions, "session_id": session_id}
        
        else:
            return {"message": "command not supported", "result": None}

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