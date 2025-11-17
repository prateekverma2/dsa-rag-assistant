import os
import json
import re
from dataclasses import dataclass
from typing import Literal, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


Verdict = Literal["APPROVE", "REJECT", "UNSURE"]


@dataclass
class ValidationResult:
    verdict: Verdict
    score: float
    reasoning: str
    revised_answer: Optional[str] = None


class DualModelValidator:
    """
     Dual Validator supporting both Groq & OpenAI backends.
    - Prefers Groq (fast, free)
    - Falls back to OpenAI automatically if Groq fails or unavailable.
    """

    def __init__(self, validator_model: Optional[str] = None):
        # Preferred provider from .env
        self.provider = os.getenv("VALIDATOR_PROVIDER", "groq").lower()
        self.model = validator_model or os.getenv("VALIDATOR_MODEL", "gpt-oss-20b")

        # --- Load API Keys ---
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # --- Initialize client ---
        self.client = None
        if self.provider == "groq" and Groq and self.groq_key:
            self.client = Groq(api_key=self.groq_key)
            print(f"✅ Validator initialized (Groq, model={self.model})")
        elif self.provider == "openai" and OpenAI and self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
            print(f"✅ Validator initialized (OpenAI, model={self.model})")
        else:
            raise RuntimeError("❌ No valid provider/API key found for validator.")

    
    def _build_prompt(self, question: str, answer: str, context: str) -> str:
        return f"""
You are a **strict DSA tutor** responsible for validating student answers.
Analyze the QUESTION, CONTEXT, and ANSWER carefully and return structured JSON.

Rules:
- "APPROVE" if the answer is conceptually correct or complete.
- "REJECT" if the answer is incorrect, incomplete, or irrelevant.
- "UNSURE" only if you cannot decide.
- Include a confidence score (0.0–1.0)
- Include a brief reasoning.
- Optionally include a corrected version as "revised_answer" if REJECT.

Return ONLY valid JSON in this exact structure:
{{
  "verdict": "APPROVE" | "REJECT" | "UNSURE",
  "score": float,
  "reasoning": "short explanation",
  "revised_answer": "optional corrected text"
}}

QUESTION:
{question}

CONTEXT:
{context or "[NO CONTEXT PROVIDED]"}

ANSWER:
{answer}
"""

   
    def verify(self, question: str, answer: str, context: str) -> ValidationResult:
        prompt = self._build_prompt(question, answer, context)

        # ✅ Try Groq first
        if self.provider == "groq":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
                raw_output = response.choices[0].message.content or "{}"
                return self._parse_response(raw_output)

            except Exception as e:
                print(f"⚠️ Groq Validator Error: {e}")
                # 🔁 Fallback to OpenAI if available
                if OpenAI and self.openai_key:
                    print("🔁 Falling back to OpenAI validator...")
                    self.provider = "openai"
                    self.client = OpenAI(api_key=self.openai_key)
                    self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    return self.verify(question, answer, context)
                else:
                    return ValidationResult("UNSURE", 0.0, f"Groq error: {e}")

        # ✅ If provider is OpenAI directly
        elif self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
                raw_output = response.choices[0].message.content or "{}"
                return self._parse_response(raw_output)

            except Exception as e:
                print(f"⚠️ OpenAI Validator Error: {e}")
                return ValidationResult("UNSURE", 0.0, f"OpenAI error: {e}")

        # ❌ No valid provider found
        return ValidationResult("UNSURE", 0.0, "No valid provider configured")

    
    def _parse_response(self, raw_output: str) -> ValidationResult:
        try:
            data = json.loads(raw_output)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw_output)
            data = json.loads(match.group(0)) if match else {
                "verdict": "UNSURE",
                "score": 0.5,
                "reasoning": "Invalid JSON output"
            }

        verdict = data.get("verdict", "UNSURE").upper()
        if verdict not in ("APPROVE", "REJECT", "UNSURE"):
            verdict = "UNSURE"

        return ValidationResult(
            verdict=verdict,
            score=float(data.get("score", 0.5)),
            reasoning=data.get("reasoning", ""),
            revised_answer=data.get("revised_answer")
        )
