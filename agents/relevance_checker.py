# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
from utils.qwen_llm import QwenLLM

class RelevanceChecker:
    def __init__(self, llm, pipe=None, tokenizer=None):
        self.llm = llm
        self.prompt_template = (
            "You are a document analysis assistant. Classify how well the given passages address the user's question.\n\n"
            "CLASSIFICATION OPTIONS:\n"
            "- CAN_ANSWER: Passages contain sufficient information to fully answer the question\n"
            "- PARTIAL: Passages contain some relevant information but lack complete details\n" 
            "- NO_MATCH: Passages do not contain any relevant information\n\n"
            "INSTRUCTIONS:\n"
            "- Respond with ONLY one word: CAN_ANSWER, PARTIAL, or NO_MATCH\n"
            "- Do not provide explanations or additional text\n"
            "- If passages mention the topic at all, choose PARTIAL (not NO_MATCH)\n\n"
            "Question: {question}\n\n"
            "Passages: {document_content}\n\n"
            "Classification:"
        )

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM chain for classification.
        
        Returns: "CAN_ANSWER" or "PARTIAL" or "NO_MATCH".
        """

        print(f"[DEBUG] RelevanceChecker.check called with question='{question}' and k={k}")
        
        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            print("[DEBUG] No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Print how many docs were retrieved in total
        print(f"[DEBUG] Retriever returned {len(top_docs)} docs. Now taking top {k} to feed LLM.")

        # Show a quick snippet of each chunk for debugging
        for i, doc in enumerate(top_docs[:k]):
            snippet = doc.page_content[:200].replace("\n", "\\n")
            print(f"[DEBUG] Chunk #{i+1} preview (first 200 chars): {snippet}...")

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])
        print(f"[DEBUG] Combined text length for top {k} chunks: {len(document_content)} chars.")

        prompt = self.prompt_template.format(question=question, document_content=document_content)
        print(f"prompt: {prompt}")
        response = self.llm.generate(prompt, max_new_tokens=50, temperature=0.1)
        print(f"[DEBUG] LLM raw classification response: '{response}'")
        
        # Cải thiện parsing: tìm label đầu tiên trong response
        classification = None
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        response_upper = response.upper().strip()
        
        # Kiểm tra từng dòng của response
        for line in response_upper.split('\n'):
            line = line.strip()
            if line in valid_labels:
                classification = line
                break
        
        # Nếu không tìm thấy exact match, tìm bằng substring
        if not classification:
            for label in valid_labels:
                if label in response_upper:
                    classification = label
                    break
        
        if not classification:
            print("[DEBUG] LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            print(f"[DEBUG] Classification recognized as '{classification}'.")
        return classification
