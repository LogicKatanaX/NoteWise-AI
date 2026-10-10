import json
import ast
import operator
from typing import Dict, Any, List
from llama_index.core import Settings
from app.engine.rag_engine import NoteWiseRAG

class NoteWiseAgent:
    """Lightweight Agentic layer over NoteWiseRAG with simple tool-calling capabilities."""
    
    def __init__(self, rag_engine: NoteWiseRAG):
        self.rag_engine = rag_engine
        # Use the global LLM configured in LlamaIndex settings
        self.llm = Settings.llm
        
    def calculator(self, expression: str) -> str:
        """Safely evaluate basic math expressions using AST."""
        allowed_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
        }
        
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num): # Fallback for older python versions
                return node.n
            elif isinstance(node, ast.BinOp):
                return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return allowed_ops[type(node.op)](_eval(node.operand))
            else:
                raise ValueError(f"Unsupported math operation: {type(node)}")
        
        try:
            # Clean up the expression
            expression = expression.strip().replace("`", "")
            tree = ast.parse(expression, mode='eval').body
            return str(_eval(tree))
        except Exception as e:
            return f"Error calculating '{expression}': {e}"
            
    def summarize_sources(self, sources: List[Any]) -> str:
        """Summarize the retrieved source nodes."""
        if not sources:
            return "No sources available to summarize."
            
        texts = [node.node.text for node in sources]
        combined_text = "\n\n---\n\n".join(texts)
        
        prompt = (
            "Please provide a concise, high-level summary of the following retrieved information:\n\n"
            f"{combined_text}"
        )
        
        try:
            response = self.llm.complete(prompt)
            return str(response)
        except Exception as e:
            return f"Error summarizing sources: {e}"

    def chat(self, question: str) -> Dict[str, Any]:
        """
        Main agent logic:
        1. Decide if we need RAG, Calculator, or Direct Answer.
        2. Execute the corresponding action.
        3. Optionally summarize RAG sources.
        """
        system_prompt = f"""
        You are the NoteWise intelligent assistant. Analyze the user's question and decide the best tool to use.
        
        Available tools:
        - "rag_engine": For querying documents, private knowledge bases, or finding information.
        - "calculator": For evaluating mathematical expressions (e.g., "5 + 5 * 10").
        - "direct": For direct conversation, general knowledge, greetings, or when no other tool is needed.
        
        Respond STRICTLY with a valid JSON object in this format:
        {{"tool": "tool_name", "args": "The exact search query, math expression, or direct question"}}
        
        User Question: "{question}"
        """
        
        # 1. Routing Logic
        tool = "rag_engine"  # Safe default
        args = question
        
        try:
            decision_str = str(self.llm.complete(system_prompt)).strip()
            # Clean markdown formatting if the LLM adds it
            if decision_str.startswith("```json"):
                decision_str = decision_str[7:-3].strip()
            elif decision_str.startswith("```"):
                decision_str = decision_str[3:-3].strip()
                
            decision = json.loads(decision_str)
            tool = decision.get("tool", "rag_engine")
            args = decision.get("args", question)
        except Exception as e:
            print(f"Failed to parse LLM router decision, falling back to RAG. Error: {e}")
            
        # 2. Execution Logic
        tools_used = []
        source_nodes = []
        answer = ""
        
        if tool == "calculator":
            tools_used.append("calculator")
            answer = self.calculator(args)
            
        elif tool == "rag_engine":
            tools_used.append("rag_engine")
            try:
                rag_res = self.rag_engine.query(args)
                answer = rag_res["answer"]
                source_nodes = rag_res.get("source_nodes", [])
                
                # 3. Post-processing Tool Check (Summarization)
                if "summarize" in question.lower() or "summary" in question.lower():
                    tools_used.append("summarize_sources")
                    answer = self.summarize_sources(source_nodes)
                    
            except ValueError as ve:
                answer = f"RAG Engine warning: {ve}"
            except Exception as e:
                answer = f"Error querying RAG engine: {e}"
                
        else:
            # Direct Answer
            tools_used.append("direct")
            try:
                answer = str(self.llm.complete(question))
            except Exception as e:
                answer = f"Error generating response: {e}"
                
        return {
            "answer": answer,
            "source_nodes": source_nodes,
            "tools_used": tools_used
        }

