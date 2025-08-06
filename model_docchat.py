import os
from typing import List, Tuple
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from utils.logging import logger

# This function will be used by Qwen3 for docchat prediction and UI

def docchat_answer(question: str, file_paths: List[str], model_name_or_path="Qwen/Qwen3-1.7B", pipe=None, tokenizer=None) -> Tuple[str, str]:
    """
    Process the given files and question using the DocChat pipeline.
    Returns (answer, verification_report).
    """
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow(model_name_or_path, pipe, tokenizer)

    # Prepare file-like objects for DocumentProcessor
    files = []
    for path in file_paths:
        if os.path.exists(path):
            class FileObj:
                def __init__(self, name):
                    self.name = name
            files.append(FileObj(path))
        else:
            logger.warning(f"File not found: {path}")

    if not files:
        return ("❌ No valid documents found.", "")

    try:
        chunks = processor.process(files)
        print(f"chunks: {chunks}")

        retriever = retriever_builder.build_hybrid_retriever(chunks)
        print(f"retriever: {retriever}")
        print(f"question: {question}")

        result = workflow.full_pipeline(question=question, retriever=retriever)
        print(f"result: {result}")
        return result["draft_answer"], result["verification_report"]
    except Exception as e:
        logger.error(f"DocChat processing error: {str(e)}")
        return (f"❌ Error: {str(e)}", "") 