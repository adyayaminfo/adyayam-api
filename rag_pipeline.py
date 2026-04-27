"""
Adyayam RAG Pipeline
Ingests CA Foundation PDFs → chunks → embeds → stores in vector DB
Run this once to build your knowledge base, then re-run when you add new PDFs
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PDFS_DIR = "./pdfs"          # Put your CA Foundation PDFs here
CHUNK_SIZE = 800             # Characters per chunk (optimal for CA content)
CHUNK_OVERLAP = 100          # Overlap between chunks to preserve context
EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI embedding model
VECTOR_DB = "pinecone"       # "pinecone" | "chroma" | "supabase"

SUBJECT_MAPPING = {
    "accounts": ["account", "accounting", "financial", "journal", "ledger", "trial", "balance sheet"],
    "law": ["law", "act", "contract", "companies", "negotiable", "llp", "sale of goods"],
    "maths": ["mathematics", "maths", "statistics", "probability", "calculus", "sets", "logical"],
    "economics": ["economics", "demand", "supply", "market", "gdp", "inflation", "rbi", "monetary"]
}

# ─────────────────────────────────────────────
# PDF PARSER
# ─────────────────────────────────────────────

class PDFParser:
    """
    Parse PDFs using LlamaParse (best for CA material with tables/formulas)
    Falls back to PyMuPDF for simple PDFs
    """
    
    def __init__(self, use_llamaparse: bool = True):
        self.use_llamaparse = use_llamaparse
    
    def parse(self, pdf_path: str) -> str:
        """Extract clean text from PDF"""
        if self.use_llamaparse:
            return self._parse_with_llamaparse(pdf_path)
        else:
            return self._parse_with_pymupdf(pdf_path)
    
    def _parse_with_llamaparse(self, pdf_path: str) -> str:
        """
        LlamaParse: Best for complex PDFs with tables, formulas
        Install: pip install llama-parse
        Get key: cloud.llamaindex.ai
        """
        try:
            from llama_parse import LlamaParse
            
            parser = LlamaParse(
                api_key=os.environ.get("LLAMAPARSE_API_KEY"),
                result_type="markdown",
                verbose=False,
                language="en",
                # CA-specific instructions
                parsing_instruction="""
                This is a CA Foundation study material PDF. 
                Preserve all:
                - Table structures (journal entries, ledger accounts, trial balances)
                - Numbered sections and sub-sections  
                - Formula notations
                - Legal section references (e.g., Section 2(1)(a))
                - Mathematical equations
                Extract text in reading order. Mark tables clearly.
                """
            )
            
            documents = parser.load_data(pdf_path)
            return "\n\n".join([doc.text for doc in documents])
            
        except ImportError:
            print("LlamaParse not installed. Install: pip install llama-parse")
            print("Falling back to PyMuPDF...")
            return self._parse_with_pymupdf(pdf_path)
    
    def _parse_with_pymupdf(self, pdf_path: str) -> str:
        """
        PyMuPDF fallback: Works for text-based PDFs
        Install: pip install pymupdf
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except ImportError:
            print("PyMuPDF not installed. Install: pip install pymupdf")
            raise
    
    def detect_subject(self, filename: str, content: str) -> str:
        """Auto-detect subject from filename and content"""
        filename_lower = filename.lower()
        content_lower = content[:2000].lower()  # Check first 2000 chars
        
        scores = {}
        for subject, keywords in SUBJECT_MAPPING.items():
            score = sum(1 for kw in keywords if kw in filename_lower or kw in content_lower)
            scores[subject] = score
        
        detected = max(scores, key=scores.get)
        return detected if scores[detected] > 0 else "general"


# ─────────────────────────────────────────────
# TEXT CHUNKER
# ─────────────────────────────────────────────

class SmartChunker:
    """
    CA-aware chunker that respects:
    - Chapter/section boundaries
    - Complete journal entry blocks
    - Full legal provision paragraphs
    - Complete solved examples
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, subject: str, source_file: str) -> List[Dict]:
        """Split text into smart chunks with metadata"""
        
        # Split on natural CA content boundaries
        sections = self._split_on_boundaries(text)
        chunks = []
        
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Split large sections with overlap
                sub_chunks = self._split_with_overlap(section)
                chunks.extend(sub_chunks)
        
        # Add metadata to each chunk
        result = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip tiny chunks
                continue
            
            result.append({
                "id": hashlib.md5(f"{source_file}_{i}_{chunk[:50]}".encode()).hexdigest(),
                "text": chunk.strip(),
                "metadata": {
                    "source": source_file,
                    "subject": subject,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                    "has_numbers": any(c.isdigit() for c in chunk),
                    "has_section_ref": any(kw in chunk.lower() for kw in ["section", "act", "rule", "schedule"]),
                    "has_formula": any(c in chunk for c in ["=", "×", "÷", "∑", "%"]),
                    "has_journal_entry": any(kw in chunk.lower() for kw in ["dr", "cr", "debit", "credit", "journal"]),
                }
            })
        
        return result
    
    def _split_on_boundaries(self, text: str) -> List[str]:
        """Split on chapter/section headers"""
        import re
        
        # CA-specific boundary patterns
        patterns = [
            r'\n(?=Chapter\s+\d+)',
            r'\n(?=CHAPTER\s+\d+)',  
            r'\n(?=\d+\.\d+\s+[A-Z])',  # Numbered sections like "1.1 Introduction"
            r'\n(?=ILLUSTRATION\s+\d+)',
            r'\n(?=EXAMPLE\s+\d+)',
            r'\n(?=SOLVED\s+EXAMPLE)',
            r'\n(?=PRACTICE\s+QUESTION)',
        ]
        
        combined_pattern = '|'.join(patterns)
        sections = re.split(combined_pattern, text)
        return [s for s in sections if s.strip()]
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """Split large text with overlap to preserve context"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good break point (paragraph > sentence > word)
            break_point = text.rfind('\n\n', start, end)
            if break_point == -1 or break_point == start:
                break_point = text.rfind('. ', start, end)
            if break_point == -1 or break_point == start:
                break_point = text.rfind(' ', start, end)
            if break_point == -1:
                break_point = end
            
            chunks.append(text[start:break_point + 1])
            start = break_point + 1 - self.overlap
        
        return chunks


# ─────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────

class Embedder:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts"""
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Process in batches of 100
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            response = client.embeddings.create(
                input=batch,
                model=self.model
            )
            
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            
            time.sleep(0.1)  # Rate limit courtesy
        
        return all_embeddings


# ─────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:
    """Pinecone vector store for production use"""
    
    def __init__(self):
        self.index_name = "adyayam-ca-foundation"
        self.dimension = 1536  # text-embedding-3-small dimension
    
    def setup(self):
        """Initialize Pinecone index"""
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        if self.index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(10)  # Wait for index to be ready
        
        self.index = pc.Index(self.index_name)
        print(f"✅ Connected to Pinecone index: {self.index_name}")
        return self
    
    def upsert(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Insert chunks with embeddings into Pinecone"""
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"][:1000]  # Pinecone metadata limit
                }
            })
        
        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            self.index.upsert(vectors=batch)
            print(f"  Upserted {min(i+100, len(vectors))}/{len(vectors)} vectors")
    
    def query(self, query_embedding: List[float], subject: str = None, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        filter_dict = {}
        if subject and subject != "general":
            filter_dict = {"subject": {"$eq": subject}}
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        return [
            {
                "text": match.metadata.get("text", ""),
                "subject": match.metadata.get("subject", ""),
                "source": match.metadata.get("source", ""),
                "score": match.score
            }
            for match in results.matches
            if match.score > 0.7  # Only high-confidence matches
        ]


class ChromaVectorStore:
    """
    ChromaDB for local development (no API key needed)
    Install: pip install chromadb
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
    
    def setup(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="adyayam_ca_foundation",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ ChromaDB initialized at {self.persist_dir}")
        return self
    
    def upsert(self, chunks: List[Dict], embeddings: List[List[float]]):
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    
    def query(self, query_embedding: List[float], subject: str = None, top_k: int = 5) -> List[Dict]:
        where = {"subject": subject} if subject else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        return [
            {
                "text": doc,
                "subject": meta.get("subject", ""),
                "source": meta.get("source", ""),
                "score": 1 - dist  # Convert distance to similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]


# ─────────────────────────────────────────────
# RETRIEVER (used by main.py at query time)
# ─────────────────────────────────────────────

class Retriever:
    """Retrieves relevant context for a student query"""
    
    def __init__(self, vector_store, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, subject: str = None, top_k: int = 5) -> str:
        """Get relevant context for a query"""
        # Embed the query
        query_embedding = self.embedder.embed([query])[0]
        
        # Get relevant chunks
        chunks = self.vector_store.query(query_embedding, subject, top_k)
        
        if not chunks:
            return ""
        
        # Format context for LLM
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source {i+1}: {chunk['source']} | {chunk['subject']}]\n{chunk['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)


# ─────────────────────────────────────────────
# INGESTION PIPELINE — Run this to load your PDFs
# ─────────────────────────────────────────────

def ingest_pdfs(pdfs_dir: str = PDFS_DIR, use_chroma: bool = True):
    """
    Main ingestion pipeline.
    
    HOW TO USE:
    1. Put all your CA Foundation PDFs in the ./pdfs folder
    2. Set your API keys in environment variables
    3. Run: python rag_pipeline.py
    
    Args:
        pdfs_dir: Directory containing your PDFs
        use_chroma: True = local ChromaDB (dev), False = Pinecone (production)
    """
    print("🚀 Adyayam RAG Pipeline — Starting ingestion")
    print("=" * 60)
    
    # Initialize components
    parser = PDFParser(use_llamaparse=bool(os.environ.get("LLAMAPARSE_API_KEY")))
    chunker = SmartChunker()
    embedder = Embedder()
    
    # Setup vector store
    if use_chroma:
        vector_store = ChromaVectorStore().setup()
    else:
        vector_store = VectorStore().setup()
    
    # Find all PDFs
    pdf_files = list(Path(pdfs_dir).glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDFs found in {pdfs_dir}")
        print("   Create a ./pdfs folder and add your CA Foundation PDFs")
        print("\n   Suggested folder structure:")
        print("   ./pdfs/accounts/chapter1_basic_accounting.pdf")
        print("   ./pdfs/law/indian_contract_act.pdf")
        print("   ./pdfs/maths/chapter1_ratio.pdf")
        print("   ./pdfs/economics/chapter1_demand.pdf")
        return
    
    print(f"📚 Found {len(pdf_files)} PDFs to process")
    
    total_chunks = 0
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        
        # Parse
        print("  → Parsing PDF...")
        text = parser.parse(str(pdf_path))
        print(f"  → Extracted {len(text):,} characters")
        
        # Detect subject
        subject = parser.detect_subject(pdf_path.name, text)
        print(f"  → Detected subject: {subject}")
        
        # Chunk
        chunks = chunker.chunk(text, subject, pdf_path.name)
        print(f"  → Created {len(chunks)} chunks")
        
        if not chunks:
            print("  ⚠️  No chunks created, skipping")
            continue
        
        # Embed
        print(f"  → Generating embeddings...")
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        
        # Store
        print(f"  → Storing in vector DB...")
        vector_store.upsert(chunks, embeddings)
        
        total_chunks += len(chunks)
        print(f"  ✅ Done! {len(chunks)} chunks indexed")
    
    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete!")
    print(f"   Total PDFs processed: {len(pdf_files)}")
    print(f"   Total chunks stored: {total_chunks:,}")
    print(f"   Vector DB: {'ChromaDB (local)' if use_chroma else 'Pinecone (cloud)'}")
    print("\n   Your knowledge base is ready. Start the API with:")
    print("   uvicorn backend.main:app --reload")


def check_stats():
    """Check what's in your vector database"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("adyayam_ca_foundation")
        count = collection.count()
        print(f"📊 ChromaDB Stats:")
        print(f"   Total chunks: {count:,}")
        
        # Sample by subject
        for subject in ["accounts", "law", "maths", "economics"]:
            results = collection.get(where={"subject": subject}, limit=1)
            count_results = collection.count()
            print(f"   Subject '{subject}': available")
            
    except Exception as e:
        print(f"Could not connect to ChromaDB: {e}")


if __name__ == "__main__":
    import sys
    import gc

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        check_stats()
    else:
        print("Running in batch mode (5 PDFs at a time)...")
        parser = PDFParser(use_llamaparse=False)
        chunker = SmartChunker()
        vector_store = ChromaVectorStore().setup()
        
        pdf_files = list(Path(PDFS_DIR).glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDFs")
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"[{i+1}/{len(pdf_files)}] {pdf_path.name}")
                text = parser.parse(str(pdf_path))
                subject = parser.detect_subject(pdf_path.name, text)
                chunks = chunker.chunk(text, subject, pdf_path.name)
                if chunks:
                    dummy_embeddings = [[0.0] * 1536] * len(chunks)
                    vector_store.upsert(chunks, dummy_embeddings)
                    print(f"  ✅ {len(chunks)} chunks stored")
                gc.collect()
            except Exception as e:
                print(f"  ⚠️  Skipped: {e}")
                continue
        print("\n✅ All PDFs processed!")
