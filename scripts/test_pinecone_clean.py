from dotenv import load_dotenv
from pinecone import Pinecone
import os, sys

load_dotenv()
print("Python:", sys.version)
print("Pinecone module:", Pinecone)
print("Pinecone path:", sys.modules["pinecone"].__file__)
print("Key prefix:", os.getenv("PINECONE_API_KEY")[:8])

try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("list_indexes():", pc.list_indexes())
    print("✅ Everything works inside build context too.")
except Exception as e:
    print("❌ Exception:", type(e).__name__, "→", e)
