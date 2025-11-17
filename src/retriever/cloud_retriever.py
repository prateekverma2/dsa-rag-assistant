import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()


class Embeddings384:
    """384-D embeddings using sentence-transformers/all-MiniLM-L6-v2."""
    def __init__(self, model_name: Optional[str] = None):
        name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(name)

    def embed(self, text: str) -> List[float]:
        v = self.model.encode(text, normalize_embeddings=True)
        return v.tolist()


class CloudRetriever:
    """
    Pinecone retriever that supports:
      - search_chunks(query, top_k)
      - find_qa_answer(question, threshold)
      - upsert_qa(question, answer)
    Assumes:
      - <namespace>: document chunks
      - <namespace>-qa: question/answer pairs
    """

    def __init__(self, namespace: Optional[str] = None):
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "main")
        self.qa_namespace = f"{self.namespace}-qa"
        self.index_name = os.getenv("PINECONE_INDEX")
        if not self.index_name:
            raise RuntimeError("PINECONE_INDEX not set in .env")

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set in .env")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(self.index_name)
        self.embedder = Embeddings384()

    # ---------- internal ----------

    def _vec(self, text: str) -> List[float]:
        return self.embedder.embed(text)

    def _format_match(self, m) -> Dict[str, Any]:
        # m is dict-like: {id, score, metadata}
        md = m.get("metadata", {}) if isinstance(m, dict) else {}

        text = (
            md.get("text")
            or md.get("chunk")
            or md.get("chunk_text")
            or md.get("content")
            or md.get("page_text")
            or ""
        )

        source = md.get("source") or md.get("file") or md.get("filename") or "unknown"

        return {
            "id": m.get("id"),
            "score": m.get("score"),
            "text": text,
            "source": source,
            "metadata": md,
        }



    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search both main and QA namespaces for better recall."""
        vec = self._vec(query)
        print(f"🧩 Querying Pinecone index='{self.index_name}', namespace='{self.namespace}' (top_k={top_k})")

       
        res_main = self.index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )

      
        qa_namespace = f"{self.namespace}-qa"
        res_qa = self.index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=qa_namespace
        )

        
        def extract_matches(res):
            if hasattr(res, "matches"):
                return res.matches
            elif isinstance(res, dict) and "matches" in res:
                return res["matches"]
            elif isinstance(res, dict) and "results" in res:
                return res["results"][0].get("matches", [])
            else:
                return []

        matches_main = extract_matches(res_main)
        matches_qa = extract_matches(res_qa)

     
        matches = matches_main + matches_qa
        matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        print(f"📊 Matches retrieved: {len(matches)} (main + qa combined)")
        return [self._format_match(m) for m in matches]

    def find_qa_answer(self, question: str, threshold: float = 0.50, top_k: int = 3) -> Optional[Dict[str, Any]]:
        vec = self._vec(question)
        res = self.index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=self.qa_namespace,
        )


        qa_res = self.index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=f"{self.namespace}-qa"
        )

    
        matches = (res.get("matches", []) if isinstance(res, dict) else []) + \
                  (qa_res.get("matches", []) if isinstance(qa_res, dict) else [])

        if not matches:
            return None

        top = matches[0]
        score = top.get("score", 0.0)
        if score < threshold:
            return None

        md = top.get("metadata", {})
        return {
            "id": top.get("id"),
            "score": score,
            "question": md.get("question"),
            "answer": md.get("answer"),
            "source": md.get("source", "qa_index"),
        }

    def upsert_qa(self, question: str, answer: str, source: str = "autolearn") -> None:
        """Insert a QA pair into the QA namespace."""
        qv = self._vec(question)
        self.index.upsert(
            vectors=[
                {
                    "id": f"qa::{abs(hash(question))}",
                    "values": qv,
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "source": source,
                    },
                }
            ],
            namespace=self.qa_namespace,
        )



if __name__ == "__main__":
    import sys
    import traceback

    print("\n🔍 Testing Retriever...\n")

    try:
        retriever = CloudRetriever(namespace="main")

        # User query
        query = input("👨‍🎓 Enter a test query (default: 'Explain binary search algorithm'): ").strip() \
                or "Explain binary search algorithm"

        print(f"\n🔹 Searching top chunks for: {query}")
        chunks = retriever.search_chunks(query, top_k=3)

        if not chunks:
            print("⚠️ No chunks found (empty index or wrong namespace).")
        else:
            for i, c in enumerate(chunks, 1):
                src = c.get("source", "unknown")
                text_preview = c.get("text", "")[:120].replace("\n", " ")
                score = c.get("score", 0.0)
                print(f"{i}. [{src}] (score={score:.2f}) {text_preview}...")

        print("\n🔹 Checking QA index hit...")
        qa = retriever.find_qa_answer(query, threshold=0.50)

        if qa:
            print(f"✅ Found QA hit (score={qa['score']:.2f})")
            print(f"💬 Answer: {qa['answer'][:200]}...\n")
        else:
            print("❌ No QA hit found — ready for LLM generation or indexing test.\n")

        # Optional QA insertion
        test_insert = input("🧩 Insert a demo QA pair? (y/n): ").strip().lower()
        if test_insert == "y":
            q = input("Enter question: ").strip() or "What is Bubble Sort?"
            a = input("Enter answer: ").strip() or "Bubble Sort is a simple comparison-based algorithm."
            retriever.upsert_qa(q, a)
            print("✅ QA inserted successfully!")

        print("\n🎯 Test completed successfully.\n")

    except KeyboardInterrupt:
        print("\n🛑 Test cancelled by user.")
    except Exception as e:
        print("❌ Error during retriever test:")
        traceback.print_exc(file=sys.stdout)