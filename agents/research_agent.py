# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from utils.qwen_llm import QwenLLM
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, llm, pipe=None, tokenizer=None):
        """Initialize the research agent with the OpenAI model."""
        self.llm = llm
        self.prompt_template = (
            "You are a helpful assistant. Answer the question based on the provided context.\n\n"
            "Instructions:\n"
            "- For summary requests: Provide a comprehensive summary of the main points\n"
            "- For specific questions: Answer precisely using the context\n"
            "- Be factual and use information from the context\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )
        
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = self.prompt_template.format(question=question, context=context)
        try:
            answer = self.llm.generate(prompt, max_new_tokens=512, temperature=0.7)
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Context used: {context[:500]}...")  # Limit context logging
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
        
        return {
            "draft_answer": answer,
            "context_used": context
        }