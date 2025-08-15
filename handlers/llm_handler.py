import asyncio
from typing import Dict, Any
from language_model.plugin_loader import *
from language_model.llm_base import create_plugin
from entities import ActionRequest


# Global storage for LLM plugins
active_llm_plugins: Dict[str, Any] = {}

def initialize_llm_plugin(agent_name: str, llm_config: Dict[str, Any]) -> bool:
    """Initialize LLM plugin for an agent"""
    try:
        llm_plugin = create_plugin(
            llm_config["plugin_type"],
            **llm_config["config_model"],
        ).load()
        active_llm_plugins[agent_name] = llm_plugin
        print(f"[LLM] LLM plugin loaded for agent: {agent_name}")
        return True
    except Exception as e:
        print(f"[LLM] Failed to initialize LLM plugin for {agent_name}: {e}")
        return False

async def generate_ai_response_with_plugin(
    text: str, agent_name: str, session_id: str, model_instance
) -> str:
    """Generate AI response using model or plugin"""
    try:
        # Check if agent has a specific LLM plugin
        llm_plugin = active_llm_plugins.get(agent_name)
        
        if llm_plugin:
            # Use agent-specific LLM plugin
            result = await asyncio.to_thread(llm_plugin.predict, text)
            
            if isinstance(result, dict):
                return result.get("response") or result.get("text") or "I'm sorry, I couldn't process that."
            else:
                return str(result) if result else "I'm sorry, I couldn't process that."
        else:
            # Use default model instance            
            action_request = ActionRequest(
                command="predict",
                params={
                    "prompt": text,
                    "enable_function_calling": False,
                },
                session_id=session_id,
                use_history=True
            )

            response = model_instance.action(
                action_request.command,
                **action_request.params,
                session_id=session_id,
                use_history=action_request.use_history
            )
            
            # Extract text from response
            if isinstance(response, dict):
                return response.get("response") or response.get("text") or "I'm sorry, I couldn't process that."
            else:
                return str(response) if response else "I'm sorry, I couldn't process that."
        
    except Exception as e:
        print(f"[LLM] Response generation error: {e}")
        return "I'm sorry, there was an error processing your request."

async def generate_ai_response(text: str, agent_name: str, session_id: str, model_instance) -> str:
    """Generate AI response using model (fallback function for compatibility)"""
    try:
        # Use existing model to generate response        
        action_request = ActionRequest(
            command="predict",
            params={
                "prompt": text,
                "enable_function_calling": False,
                "agent_name": agent_name,
            },
            session_id=session_id,
            use_history=True
        )

        response = model_instance.action(
            action_request.command,
            **action_request.params,
            session_id=session_id,
            use_history=action_request.use_history
        )
        
        # Extract text from response
        if isinstance(response, dict):
            return response.get("response") or response.get("text") or "I'm sorry, I couldn't process that."
        else:
            return str(response) if response else "I'm sorry, I couldn't process that."
        
    except Exception as e:
        print(f"[AI] Response generation error: {e}")
        return "I'm sorry, there was an error processing your request."

def get_active_llm_plugins():
    """Get all active LLM plugins"""
    return active_llm_plugins

def get_llm_plugin(agent_name: str):
    """Get LLM plugin for a specific agent"""
    return active_llm_plugins.get(agent_name)

def cleanup_llm_plugin(agent_name: str) -> bool:
    """Clean up LLM plugin for an agent"""
    try:
        if agent_name in active_llm_plugins:
            del active_llm_plugins[agent_name]
            print(f"[LLM] Cleaned up LLM plugin for agent: {agent_name}")
            return True
        return False
    except Exception as e:
        print(f"[LLM] Failed to cleanup LLM plugin for {agent_name}: {e}")
        return False