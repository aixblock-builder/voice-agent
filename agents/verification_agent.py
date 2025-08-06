# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
# from config.settings import settings
from utils.qwen_llm import QwenLLM
import logging

logger = logging.getLogger(__name__)

class VerificationAgent:
    def __init__(self, llm, pipe=None, tokenizer=None):
        self.llm = llm
        self.prompt_template = (
            "Verify the following answer against the provided context. Check for:\n"
            "1. Direct factual support (YES/NO)\n"
            "2. Unsupported claims (list)\n"
            "3. Contradictions (list)\n"
            "4. Relevance to the question (YES/NO)\n"
            "Respond in this format:\n"
            "Supported: YES/NO\n"
            "Unsupported Claims: [items]\n"
            "Contradictions: [items]\n"
            "Relevant: YES/NO\n"
            "Answer: {answer}\n"
            "Context: {context}"
        )
        
    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # chain = self.prompt | self.llm | StrOutputParser()
        # try:
        #     verification = chain.invoke({
        #         "answer": answer,
        #         "context": context
        #     })
        #     logger.info(f"Verification report: {verification}")
        #     logger.info(f"Context used: {context}")
        # except Exception as e:
        #     logger.error(f"Error verifying answer: {e}")
        #     raise
        # 
        # return {
        #     "verification_report": verification,
        #     "context_used": context
        # }
        prompt = self.prompt_template.format(answer=answer, context=context)
        try:
            verification = self.llm.generate(prompt)
            logger.info(f"Verification report: {verification}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error verifying answer: {e}")
            raise
        return {
            "verification_report": verification,
            "context_used": context
        }