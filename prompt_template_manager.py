from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from datetime import datetime

from mcp_tools_manager import MCP_TOOLS_REGISTRY


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