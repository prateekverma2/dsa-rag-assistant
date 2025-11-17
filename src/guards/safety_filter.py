import re

class SafetyFilter:
    """
     Intelligent Safety & Relevance Filter
   
    Allows any programming / LeetCode / DSA related queries — even if they mention sports etc.
     Blocks unrelated, personal, or unsafe topics.
    """

    def __init__(self):
        self.dsa_terms = [
            "array", "matrix", "stack", "queue", "linked list", "graph", "tree", "heap",
            "bfs", "dfs", "recursion", "dp", "dynamic programming", "backtracking",
            "sorting", "merge sort", "quick sort", "binary search", "greedy", "hashmap",
            "time complexity", "space complexity", "big o", "algorithm", "data structure",
            "problem", "code", "python", "function", "class", "leetcode", "codeforces",
            "gfg", "runtime", "complexity", "searching", "implementation", "solution"
        ]

        # words that may appear in non-DSA chats
        self.blocked_clusters = [
            "father", "mother", "papa", "mom", "friend", "love", "relationship",
            "movie", "song", "actor", "actress", "politics", "news",
            "porn", "sex", "nude", "weather", "match", "game", "who is", "where is",
            "girlfriend", "boyfriend", "hate", "angry", "married"
        ]

    # --------------------------------------------------------------------
    def check_question(self, question: str, context: str) -> dict:
        q = (question or "").strip().lower()

        if not q:
            return self._reject("empty", "⚠️ Please ask a valid DSA or programming question.")

        # ✅ If query explicitly has programming / algorithm / LeetCode cues → always allow
        if re.search(r"(leetcode|gfg|codeforces|code|algorithm|program|data structure|complexity|problem|python|c\+\+|solution)", q):
            return {"allow": True, "reason": "coding_context", "message": ""}

        # ✅ If DSA keyword appears → allow
        if any(k in q for k in self.dsa_terms):
            return {"allow": True, "reason": "dsa_related", "message": ""}

        # 🚫 Block only if non-DSA and lacks technical cues
        if any(b in q for b in self.blocked_clusters) and not re.search(r"(leetcode|code|algorithm|problem|solution|program)", q):
            return self._reject(
                "non_dsa_topic",
                "⚠️ I can only help with programming, algorithms, or data structure-related questions."
            )

        # 🚫 If question too vague and no DSA hints
        if len(q.split()) < 3 and not re.search(r"(code|algorithm|program|complexity|data structure|problem)", q):
            return self._reject(
                "too_vague",
                "⚠️ Please clarify your query — try something like 'Explain recursion with an example.'"
            )

        # Weak or no context → allow but mark low confidence
        if context.strip().upper() in ("", "[NO_CONTEXT]", "NO_CONTEXT"):
            return {"allow": True, "reason": "weak_context", "message": ""}

        return {"allow": True, "reason": "safe", "message": ""}

   
    @staticmethod
    def _reject(reason: str, message: str) -> dict:
        return {"allow": False, "reason": reason, "message": message}
