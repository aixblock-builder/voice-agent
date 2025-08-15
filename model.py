import json
import os
from typing import Any, Dict, Iterator, List, Optional, Union

import gradio as gr
from knowleadge_base_manager import KNOWLEDGE_BASES, KnowledgeBaseManager
from language_model.llm_base import LlmBase
from mcp_tools_manager import MCP_TOOLS_REGISTRY, MCPToolsManager
from prompt_template_manager import CUSTOM_TEMPLATES, PromptTemplateManager
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
from handlers.llm_handler import active_llm_plugins
from pydantic import BaseModel


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

class ActionRequest(BaseModel):
    command: str
    params: Dict[str, Any]
    doc_file_urls: Optional[Union[str, List[str]]] = None
    session_id: Optional[str] = None
    use_history: Optional[bool] = True

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
            agent_name = kwargs.get("agent_name", "default_agent")

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
                    global pipe_prediction, tokenizer
                    
                    if pipe_prediction:
                        return
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Determine source
                    if hf_token:
                        login(token=hf_token)
                    llm_plugin = active_llm_plugins.get(agent_name, active_llm_plugins.get(agent_name))
                    llm_plugin.load()
                    if not llm_plugin:
                        raise ValueError(f"LLM Plugin {agent_name} not found")
                    tokenizer = llm_plugin.tokenizer
                    pipe_prediction = llm_plugin.pipeline
                    
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