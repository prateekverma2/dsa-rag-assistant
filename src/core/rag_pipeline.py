import os
import time
import json
import atexit
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from src.guards.safety_filter import SafetyFilter


from dotenv import load_dotenv
load_dotenv()

# ---- Cloud / Local clients ----
from src.retriever.cloud_retriever import CloudRetriever
from src.autolearn.validator import DualModelValidator
from src.autolearn.queue import LearnQueue
from src.autolearn.promote import PromoteLearner

# Groq (preferred cloud)
try:
    from groq import Groq
except Exception:
    Groq = None

# Local (Ollama)
try:
    from ollama import chat as ollama_chat
except Exception:
    ollama_chat = None


class RAGPipeline:
    """
    Flow:
      1) QA-first lookup in Pinecone (threshold >= 0.50) → direct answer if hit
      2) Else retrieve chunks for context
      3) Generate with Cloud (Groq) OR Local (Ollama)
      4) Validate with single validator model (Groq preferred, OpenAI fallback)
      5) APPROVE → promote + index QA; REJECT/UNSURE → queue
      6) Memory save/load/delete; last 3 turns included in prompt for 1:1 conversation feel
    """

    def __init__(self,
                 mode: str = os.getenv("RAG_MODE", "cloud"),      # "cloud" | "local"
                 cloud_model: str = os.getenv("CLOUD_MODEL", "openai/gpt-oss-120b"),
                 local_model: str = os.getenv("LOCAL_MODEL", "llama3"),
                 top_k: int = int(os.getenv("TOP_K", 5)),
                 qa_threshold: float = float(os.getenv("QA_HIT_THRESHOLD", 0.50)),
                 namespace: str = os.getenv("PINECONE_NAMESPACE", "main"),
                 chat_dir: str = os.getenv("CHAT_DIR", "chat_history")):

        self.mode = mode.lower().strip()
        self.cloud_model = cloud_model
        self.local_model = local_model
        self.top_k = top_k
        self.qa_threshold = qa_threshold
        self.namespace = namespace

        self.retriever = CloudRetriever(namespace=self.namespace)
        self.validator = DualModelValidator()
        self.queue = LearnQueue()
        self.promoter = PromoteLearner()
        self.safety = SafetyFilter()

        # Cloud client (Groq)
        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY")) if Groq and os.getenv("GROQ_API_KEY") else None

        # Memory
        self.chat_history: List[Dict[str, str]] = []
        self.chat_dir = chat_dir
        os.makedirs(self.chat_dir, exist_ok=True)
        try:
            atexit.unregister(self._auto_save_memory)
        except Exception:
            pass
            atexit.register(lambda: self.           _auto_save_memory())


        print(f"🚀 RAG ready | mode={self.mode.upper()} | threshold={self.qa_threshold} | cloud={self.cloud_model} | local={self.local_model}")

  
    def _find_direct_answer(self, question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
         Similarity threshold check: if score >= 0.50 → return cached QA (no LLM cost).
        """
        try:
            res = self.retriever.find_qa_answer(question, threshold=self.qa_threshold)
            if res and res.get("answer"):
                meta = {
                    "source": res.get("source"),
                    "score": res.get("score"),
                    "id": res.get("id"),
                    "type": "qa_hit"
                }
                print(f"✅ QA hit (score={meta['score']}) — using Pinecone answer.")
                return res["answer"], meta
        except Exception as e:
            print(f"⚠️ QA lookup error: {e}")
        return None

 
    def _retrieve_chunks(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            return self.retriever.search_chunks(question, top_k=top_k) or []
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "[NO_CONTEXT]"
        blocks = []
        for c in chunks:
            src = c.get("source") or c.get("metadata", {}).get("source", "unknown")
            page = c.get("metadata", {}).get("page", "?")
            text = c.get("text") or c.get("metadata", {}).get("text", "")
            blocks.append(f"[src: {src} p.{page}] {text}")
        return "\n\n".join(blocks)

    def _build_prompt(self, question: str, context: str, memory: str) -> str:
        """
        Builds a natural, mentor-style prompt for the DSA Tutor.
        Keeps the full technical structure intact while making tone conversational.
        """
        coding_keys = [
            "code", "leetcode", "codeforces", "gfg", "class", "function", "implement",
            "solution", "tree", "linked list", "stack", "queue", "dp", "graph", "sort",
            "recursion", "dynamic", "algorithm", "python", "c++"
        ]
        is_code = any(k in question.lower() for k in coding_keys)

        # 🧠 Format the memory for continuity
        formatted_memory = (
            f"Here's the recent chat context between you and the student:\n\n{memory}\n\n"
            if memory.strip() else
            "You are starting a new conversation with a student about DSA topics.\n"
        )

        # 📚 Handle context smartly
        if not context or context.strip() == "[NO_CONTEXT]":
            context_info = (
                "No external context retrieved. Use your DSA expertise, reasoning, and intuition to guide the student clearly."
            )
        else:
            context_info = context

      
        if is_code:
            return f"""
You are a **senior software engineer and DSA mentor** helping a student understand complex algorithmic coding problems.

Be conversational and natural — respond like a human tutor who remembers the previous discussion.
Encourage, correct gently, and keep the technical explanation structured but friendly.

{formatted_memory}

---
**QUESTION (from student):**
{question}

**RETRIEVED CONTEXT (from Pinecone):**
{context_info}

---

Now continue the discussion naturally, referencing the student's previous understanding if relevant.

Your answer must include these sections (in order):

1. 👋 **Conversational Intro:** Greet and connect to previous discussion (if any).  
2. 📘 **Problem Overview:** Explain what the question is really about.  
3. ⚙️ **Approach (Step-by-Step):** Walk through your reasoning clearly.  
4. 💻 **Python Code:** Use markdown fenced code blocks (```python ... ```).  
5. ✅ **Correctness Argument:** Why this approach works logically.  
6. ⏱️ **Time & Space Complexity:** Analyze precisely.  
7. 🧩 **Key Insights:** Short summary or alternative ways to think about it.  
8. 🎯 **Motivational Closing:** One line of mentor-style advice to reinforce learning.

Keep tone warm, engaging, and slightly conversational while maintaining technical precision.
""".strip()

        else:
            return f"""
You are a **friendly DSA tutor** who teaches theoretical and conceptual questions in a conversational, easy-to-follow way.

Avoid robotic explanations — instead, speak like a mentor who remembers what was last discussed.
Reference prior context when possible.

{formatted_memory}

---
**QUESTION (from student):**
{question}

**RETRIEVED CONTEXT (from Pinecone):**
{context_info}

---

Now respond naturally while following this structure:

1. 👋 **Conversational Intro:** Reference prior topic or greet the student casually.  
2. 🧠 **Concept Overview:** Explain the concept in simple, intuitive terms.  
3. 📘 **Detailed Explanation:** Step-by-step reasoning, examples, or analogies.  
4. 💻 **Code Snippet (if applicable):** Give short Python or pseudocode examples in markdown.  
5. ⏱️ **Time & Space Complexity:** Add if it makes sense.  
6. 🧩 **Key Takeaways:** 2–3 bullet points summarizing the idea.  
7. 💬 **Closing Remark:** Encourage the student or suggest what to explore next.

Maintain a mentor-like tone, clear formatting, and continuity from the previous messages.
""".strip()

   
    def retrieve_context(self, query: str, top_k: int = 3):
        """
        Retrieve top-k context chunks from Pinecone for a given query.
        Returns: (context_text, results_list)
        """
        try:
            print(f"🔎 Retrieving top {top_k} chunks for query: {query}")
            results = self.retriever.search_chunks(query, top_k=top_k)

            if not results:
                print("⚠️ No chunks retrieved — retriever returned empty result.")
                return "", []

            # Join retrieved chunks as combined context
            context = "\n\n".join([
                f"[{r.get('source', 'unknown')}] {r.get('text', '')[:500]}..."
                for r in results
            ])

            print(f"✅ Retrieved {len(results)} chunks from Pinecone.")
            return context, results

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return "", []

    
    def generate_answer(self, query: str, context: str):
        """
        Generate a detailed answer using the current mode (cloud/local).
        Uses self.mode to decide between Groq and Ollama.
        """
        import time

        
        memory = "\n".join(
            [f"User: {h['query']}\nTutor: {h['answer']}" for h in self.chat_history[-3:]]
        )

        # Build full prompt for the LLM
        prompt = self._build_prompt(query, context, memory)

        #  If no context, add to learn queue for later improvement
        if not context:
            print("⚠️ No context found — switching to self-generation mode.")
            try:
                self.queue.add(query, "", reason="no_context_found")
            except Exception:
                pass

     
        if self.mode == "cloud":
            try:
                if not self._groq:
                    return "⚠️ Groq client not initialized. Check GROQ_API_KEY."

                start = time.time()
                response = self._groq.chat.completions.create(
                    model=self.cloud_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=800,
                )

                answer = response.choices[0].message.content.strip()
                print(f"⚡ Cloud model response in {round(time.time() - start, 2)}s")
                return answer

            except Exception as e:
                return f"❌ Cloud generation failed: {e}"


        elif self.mode == "local":
            try:
                from ollama import chat as ollama_chat
            except ImportError:
                return "⚠️ Ollama not installed — please install it or switch to cloud mode."

            try:
                start = time.time()
                response = ollama_chat(model=os.getenv("LOCAL_MODEL", "llama3"),
                                       messages=[{"role": "user", "content": prompt}])
                answer = response["message"]["content"].strip()
                print(f"⚡ Local model response in {round(time.time() - start, 2)}s")
                return answer

            except Exception as e:
                return f"❌ Local model error: {e}"


        else:
            return f"⚠️ Invalid mode '{self.mode}' — must be 'cloud' or 'local'."

    def _generate(self, question: str, context: str) -> str:
        memory = "\n".join([f"User: {h['query']}\nTutor: {h['answer']}" for h in self.chat_history[-3:]])
        prompt = self._build_prompt(question, context, memory)

        if self.mode == "cloud":
            if not self._groq:
                return "⚠️ Groq client not available (set GROQ_API_KEY)."
            try:
                t0 = time.time()
                resp = self._groq.chat.completions.create(
                    model=self.cloud_model,
                    messages=[
                        {"role": "system", "content": "You are DSA Tutor. Be precise, avoid hallucination."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                print(f"⚡ Cloud generation (Groq): {round(time.time()-t0, 2)}s")
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                return f"❌ Groq error: {e}"

        elif self.mode == "local":
            if not ollama_chat:
                return "⚠️ Ollama not installed. Switch to cloud mode."
            try:
                t0 = time.time()
                r = ollama_chat(model=self.local_model, messages=[{"role": "user", "content": prompt}])
                print(f"⚡ Local generation (Ollama): {round(time.time()-t0, 2)}s")
                return r.get("message", {}).get("content", "").strip()
            except Exception as e:
                return f"❌ Ollama error: {e}"

        return "❌ Invalid mode. Use 'cloud' or 'local'."


    def _validate(self, question: str, answer: str, context: str):
        try:
            res = self.validator.verify(question, answer, context)
            return res.verdict.lower(), res.score, res.reasoning, res.revised_answer
        except Exception as e:
            print(f"⚠️ Validator error: {e}")
            return "unsure", None, "validation_failed", None




    def _index_qa(self, question: str, answer: str):
        try:
            import hashlib
            qid = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
            vector_id = f"qa::{qid}"

            qv = self.retriever._vec(question)
            self.retriever.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": qv,
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "source": "autolearn",
                    },
                }],
            namespace=self.retriever.qa_namespace,
        )
            print(f"📥 QA stored → {vector_id} in {self.retriever.qa_namespace}")
        except Exception as e:
            print(f"⚠️ QA indexing failed: {e}")


   
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for DSA Tutor RAG System.
        - Checks cached QA (Pinecone)
        - Retrieves context chunks
        - Generates answer (Groq/Ollama)
        - Conditionally validates only when context missing
        - Auto-learns only when validator approves
        """
              
        safety_check = self.safety.check_question(question, "")
        if not safety_check["allow"]:
            print(f"🚫 Blocked by SafetyFilter → {safety_check['reason']}")
            answer = safety_check["message"]
            self._remember(question, answer)
            return {
                "mode": self.mode,
                "answer": answer,
                "verdict": "reject",
                "score": 0.0,
                "reason": safety_check["reason"],
                "learn_status": "REJECTED → SKIPPED",
                "context_chunks": [],
                "validated": False
            }
   
        qa_hit = self._find_direct_answer(question)
        if qa_hit:
            answer, meta = qa_hit
            self._remember(question, answer)
            print("✅ Cached QA hit found — skipping validation.")
            return {
                "mode": "QA_HIT",
                "answer": answer,
                "meta": meta,
                "verdict": "approve",
                "score": 1.0,
                "reason": "Answer retrieved directly from Pinecone cache",
                "learn_status": "CACHED → APPROVED",
                "context_chunks": [],
                "validated": False  
            }

     
        chunks = self._retrieve_chunks(question, top_k=self.top_k)
        context = self._format_context(chunks)
                
        safety_check = self.safety.check_question(question, context)
        if not safety_check["allow"]:
            print(f"⚠️ Blocked (weak context) → {safety_check['reason']}")
            answer = safety_check["message"]
            self._remember(question, answer)
            return {
                "mode": self.mode,
                "answer": answer,
                "verdict": "reject",
                "score": 0.0,
                "reason": safety_check["reason"],
                "learn_status": "REJECTED → NO_CONTEXT",
                "context_chunks": chunks,
                "validated": False
            }

     
        answer_raw = self._generate(question, context)

       
        avg_score = 0.0
        if chunks:
            try:
                avg_score = sum(float(c.get("score", 0.0)) for c in chunks) / len(chunks)
            except Exception:
                avg_score = 0.0

       
        context_clean = (context or "").strip().upper()
        weak_context = (not chunks) or (avg_score < 0.55) or (context_clean in ("", "[NO_CONTEXT]", "NO_CONTEXT"))

        if weak_context:
            print(f"🧪 Weak or no context detected (avg_score={avg_score:.2f}) → running validator...")
            verdict, score, reasoning, revised = self._validate(question, answer_raw, context)
            validator_raw = {
                "verdict": verdict,
                "score": score,
                "reasoning": reasoning,
                "revised_answer": revised
                 }

            final_answer = revised.strip() if (verdict == "reject" and revised) else answer_raw
            validated = True
        else:
            verdict, score, reasoning, revised = "approve", 1.0, "Context sufficient — skipping validation", None
            final_answer = answer_raw
            validated = False
            print(f"✅ Context seems sufficient (avg_score={avg_score:.2f}) — skipping validator.")

      
        if verdict == "approve":
            if validated:
          
                self.promoter.promote(
                    question,
                    final_answer,
                    meta={"score": score, "reasoning": reasoning}
                )

             
                self._index_qa(question, final_answer)

                try:
                    text_len = len(final_answer)
                    chunk_size = 800 if text_len > 800 else text_len
                    chunks_auto = [final_answer[i:i+chunk_size] for i in range(0, text_len, chunk_size)]

                    for chunk in chunks_auto:
                        self.retriever.index.upsert(
                            vectors=[{
                                "id": f"auto::{abs(hash(chunk))}",
                                "values": self.retriever._vec(chunk),
                                "metadata": {
                                    "text": chunk,
                                    "source": "autolearn",
                                    "question": question,
                                    "generated": True
                                }
                            }],
                            namespace=self.retriever.namespace
                        )
                    print(f"📘 Auto-learned new context: {len(chunks_auto)} chunks added.")
                except Exception as e:
                    print(f"⚠️ Failed to upsert chunks: {e}")
            else:
                print("✅ Context available — skipping auto-learn.")

            learn_status = "APPROVED → PROMOTED + INDEXED"

        elif verdict == "reject":
            self.queue.add(question, final_answer, reason="rejected")
            learn_status = "REJECTED → QUEUED"

        else:
            self.queue.add(question, final_answer, reason="unsure")
            learn_status = "UNSURE → QUEUED"

   
        self._remember(question, final_answer)

        return {
            "mode": self.mode,
            "answer": final_answer,
            "verdict": verdict,
            "score": score,
            "reason": reasoning,
            "learn_status": learn_status,
            "context_chunks": chunks,
            "validated": validated,
            "validator_raw": validator_raw if validated else None # type: ignore
        }

  
    def _remember(self, question: str, answer: str):
        self.chat_history.append({"query": question, "answer": answer})
        if len(self.chat_history) > 50:
            self.chat_history = self.chat_history[-50:]

    def save_chat(self, name: str) -> str:
        os.makedirs(self.chat_dir, exist_ok=True)
        path = os.path.join(self.chat_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=2)
        return path

    def load_chat(self, name: str) -> bool:
        path = os.path.join(self.chat_dir, f"{name}.json")
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self.chat_history = json.load(f)
        return True

    def delete_chat(self, name: str) -> bool:
        path = os.path.join(self.chat_dir, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def _auto_save_memory(self):
            try:
                if not getattr(self, "chat_history", None):
                    return
                os.makedirs(self.chat_dir, exist_ok=True)
                path = os.path.join(self.chat_dir, "autosave.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.chat_history, f, indent=2)
                print(f"💾 Auto-saved chat → {path}")
            except Exception as e:
                print(f"⚠️ Auto-save failed: {e}")
    
    def run_cli(self):
        print("\n🧠 DSA Tutor — Groq Cloud / Ollama Local — QA-first (threshold=0.50)")
        print("Commands: /exit | /save <name> | /load <name> | /del <name> | /mode cloud|local\n")
        while True:
            q = input("👨‍🎓 You: ").strip()
            if not q:
                continue
            if q in ("/exit", "exit", "quit"):
                break
            if q.startswith("/mode "):
                m = q.split(" ", 1)[1].strip().lower()
                if m in ("cloud", "local"):
                    self.mode = m
                    print(f"🔁 mode → {self.mode}")
                else:
                    print("⚠️ use /mode cloud|local")
                continue
            if q.startswith("/save "):
                print("💾", self.save_chat(q.split(" ", 1)[1]))
                continue
            if q.startswith("/load "):
                print("✅ loaded" if self.load_chat(q.split(" ", 1)[1]) else "⚠️ not found")
                continue
            if q.startswith("/del "):
                print("🗑️ deleted" if self.delete_chat(q.split(" ", 1)[1]) else "⚠️ not found")
                continue

            out = self.ask(q)
            print(f"\n🤖 Tutor ({out['mode']}):\n{out['answer']}\n")
            print(f"🧪 Verdict: {out['verdict']} | Learn: {out['learn_status']}")
            if out.get("context_chunks"):
                print("📚 Context sources:", [c.get("source", "?") for c in out["context_chunks"]])