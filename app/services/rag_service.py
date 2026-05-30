"""
RAG (Retrieval-Augmented Generation) Service for ASHA AI Assistant.
Uses LangChain with ChromaDB for document retrieval and Azure OpenAI embeddings.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGService:
    """Service for RAG-based document retrieval using healthcare documents."""
    
    def __init__(self, persist_directory: Optional[str] = None, docs_directory: Optional[str] = None):
        """
        Initialize RAG service with ChromaDB and Azure OpenAI embeddings.
        
        Args:
            persist_directory: Directory to persist ChromaDB
            docs_directory: Directory containing PDF documents
        """
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.docs_directory = docs_directory or settings.docs_dir
        
        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_embedding_api_version,
            azure_deployment=settings.azure_openai_embedding_deployment,
            model=settings.azure_openai_embedding_model
        )
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize or load vector store
        self.vectorstore: Optional[Chroma] = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store."""
        try:
            # Check if vector store already exists
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="asha_healthcare_docs"
                )
                logger.info(f"Loaded vector store with {self.vectorstore._collection.count()} documents")
            else:
                logger.info("No existing vector store found. Will create on first index.")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vectorstore = None
    
    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the docs directory.
        
        Returns:
            List of loaded documents
        """
        documents = []
        docs_path = Path(self.docs_directory)
        
        if not docs_path.exists():
            logger.warning(f"Documents directory not found: {self.docs_directory}")
            return documents
        
        pdf_files = list(docs_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["document_type"] = self._categorize_document(pdf_file.name)
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {e}")
        
        return documents
    
    def _categorize_document(self, filename: str) -> str:
        """Categorize document based on filename."""
        filename_lower = filename.lower()
        
        if "asha" in filename_lower or "handbook" in filename_lower:
            return "asha_handbook"
        elif "hypertension" in filename_lower:
            return "hypertension_guidelines"
        elif "imnci" in filename_lower:
            return "child_health"
        elif "sba" in filename_lower or "birth" in filename_lower:
            return "maternal_health"
        elif "guidelines" in filename_lower:
            return "general_guidelines"
        else:
            return "healthcare_reference"
    
    def index_documents(self, force_reindex: bool = False) -> int:
        """
        Index all documents in the docs directory.
        
        Args:
            force_reindex: If True, delete existing index and reindex all
        
        Returns:
            Number of document chunks indexed
        """
        try:
            # Check if reindexing is needed
            if not force_reindex and self.vectorstore is not None:
                count = self.vectorstore._collection.count()
                if count > 0:
                    logger.info(f"Vector store already has {count} documents. Skipping indexing.")
                    return count
            
            # Load documents
            documents = self.load_documents()
            
            if not documents:
                logger.warning("No documents to index")
                return 0
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
            
            # Create or recreate vector store
            if force_reindex and os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
                logger.info("Deleted existing vector store for reindexing")
            
            logger.info("Creating vector store and indexing documents...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="asha_healthcare_docs"
            )
            
            logger.info(f"Successfully indexed {len(chunks)} document chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    async def retrieve_relevant_context(
        self,
        query: str,
        k: int = 4,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_type: Optional document type filter
        
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            if self.vectorstore is None:
                logger.warning("Vector store not initialized")
                return []
            
            # Build filter if specified
            where_filter = None
            if filter_type:
                where_filter = {"document_type": filter_type}
            
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=where_filter
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "document_type": doc.metadata.get("document_type", "Unknown"),
                    "relevance_score": 1 - score  # Convert distance to similarity
                })
            
            logger.info(f"Retrieved {len(formatted_results)} relevant chunks for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    async def get_context_for_query(
        self,
        query: str,
        k: int = 4
    ) -> str:
        """
        Get formatted context string for a query.
        
        Args:
            query: The query to find context for
            k: Number of chunks to retrieve
        
        Returns:
            Formatted context string for use in prompts
        """
        context, _ = await self.get_context_with_sources(query, k)
        return context
    
    async def get_context_with_sources(
        self,
        query: str,
        k: int = 4
    ) -> tuple:
        """
        Get formatted context string and list of sources for a query.
        
        Args:
            query: The query to find context for
            k: Number of chunks to retrieve
        
        Returns:
            Tuple of (formatted context string, list of source names)
        """
        results = await self.retrieve_relevant_context(query, k)
        
        if not results:
            return ("", [])
        
        context_parts = []
        sources = []
        for i, result in enumerate(results, 1):
            source = result["source"]
            content = result["content"]
            context_parts.append(f"[Source {i}: {source}]\n{content}")
            if source not in sources:
                sources.append(source)
        
        return ("\n\n---\n\n".join(context_parts), sources)
    
    async def search_by_topic(
        self,
        topic: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search documents by healthcare topic.
        
        Args:
            topic: Healthcare topic (e.g., "hypertension", "pregnancy", "child care")
            k: Number of results
        
        Returns:
            Relevant document chunks
        """
        # Map common topics to better search queries
        topic_queries = {
            "hypertension": "high blood pressure treatment management symptoms",
            "pregnancy": "antenatal care pregnant woman prenatal checkup",
            "delivery": "skilled birth attendance labor delivery complications",
            "child": "infant child health immunization nutrition growth monitoring",
            "fever": "fever treatment management symptoms danger signs",
            "diarrhea": "diarrhea dehydration ORS treatment child",
            "malnutrition": "malnutrition nutrition feeding underweight child",
            "immunization": "vaccination immunization schedule infant child",
            "diabetes": "diabetes blood sugar management diet",
            "anemia": "anemia iron deficiency hemoglobin treatment"
        }
        
        # Use topic-specific query or the topic itself
        search_query = topic_queries.get(topic.lower(), topic)
        
        return await self.retrieve_relevant_context(search_query, k)
    
    def get_indexed_sources(self) -> set:
        """
        Get set of source files already indexed in the vector store.
        
        Returns:
            Set of source file names that are already indexed
        """
        if self.vectorstore is None:
            return set()
        
        try:
            # Get all documents from the collection to extract source metadata
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])
            
            if results and results.get("metadatas"):
                sources = set()
                for metadata in results["metadatas"]:
                    if metadata and "source_file" in metadata:
                        sources.add(metadata["source_file"])
                return sources
            return set()
        except Exception as e:
            logger.error(f"Error getting indexed sources: {e}")
            return set()
    
    def add_new_documents(self) -> Dict[str, Any]:
        """
        Add only new documents that haven't been indexed yet.
        Prevents duplicates by checking existing source files.
        
        Returns:
            Dict with counts of new documents added and skipped
        """
        try:
            docs_path = Path(self.docs_directory)
            if not docs_path.exists():
                logger.warning(f"Documents directory not found: {self.docs_directory}")
                return {"added": 0, "skipped": 0, "new_files": [], "skipped_files": []}
            
            # Get already indexed sources
            indexed_sources = self.get_indexed_sources()
            logger.info(f"Already indexed sources: {indexed_sources}")
            
            # Find new PDF files
            all_pdfs = list(docs_path.glob("*.pdf"))
            new_pdfs = [pdf for pdf in all_pdfs if pdf.name not in indexed_sources]
            skipped_pdfs = [pdf for pdf in all_pdfs if pdf.name in indexed_sources]
            
            if not new_pdfs:
                logger.info("No new documents to index")
                return {
                    "added": 0,
                    "skipped": len(skipped_pdfs),
                    "new_files": [],
                    "skipped_files": [pdf.name for pdf in skipped_pdfs]
                }
            
            logger.info(f"Found {len(new_pdfs)} new PDF files to index")
            
            # Load only new documents
            new_documents = []
            new_file_names = []
            for pdf_file in new_pdfs:
                try:
                    logger.info(f"Loading new document: {pdf_file.name}")
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source_file"] = pdf_file.name
                        doc.metadata["document_type"] = self._categorize_document(pdf_file.name)
                    
                    new_documents.extend(docs)
                    new_file_names.append(pdf_file.name)
                    logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {pdf_file.name}: {e}")
            
            if not new_documents:
                logger.warning("No documents could be loaded")
                return {"added": 0, "skipped": len(skipped_pdfs), "new_files": [], "skipped_files": [pdf.name for pdf in skipped_pdfs]}
            
            # Split new documents into chunks
            logger.info("Splitting new documents into chunks...")
            chunks = self.text_splitter.split_documents(new_documents)
            logger.info(f"Created {len(chunks)} chunks from {len(new_documents)} pages")
            
            # Add to existing vector store
            if self.vectorstore is None:
                # Create new vector store if none exists
                logger.info("Creating new vector store...")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="asha_healthcare_docs"
                )
            else:
                # Add to existing vector store
                logger.info("Adding to existing vector store...")
                self.vectorstore.add_documents(chunks)
            
            logger.info(f"Successfully added {len(chunks)} new document chunks")
            return {
                "added": len(chunks),
                "skipped": len(skipped_pdfs),
                "new_files": new_file_names,
                "skipped_files": [pdf.name for pdf in skipped_pdfs]
            }
            
        except Exception as e:
            logger.error(f"Error adding new documents: {e}")
            raise
    
    def remove_documents_by_source(self, source_file: str) -> int:
        """
        Remove all chunks from a specific source file.
        Useful for updating a document that has changed.
        
        Args:
            source_file: Name of the source file to remove
            
        Returns:
            Number of chunks removed
        """
        if self.vectorstore is None:
            return 0
        
        try:
            collection = self.vectorstore._collection
            
            # Get IDs of documents with this source
            results = collection.get(
                where={"source_file": source_file},
                include=["metadatas"]
            )
            
            if results and results.get("ids"):
                ids_to_delete = results["ids"]
                collection.delete(ids=ids_to_delete)
                logger.info(f"Removed {len(ids_to_delete)} chunks from {source_file}")
                return len(ids_to_delete)
            
            return 0
        except Exception as e:
            logger.error(f"Error removing documents: {e}")
            return 0
    
    def refresh_document(self, source_file: str) -> int:
        """
        Re-index a specific document (remove old chunks and add new).
        Use when a document has been updated.
        
        Args:
            source_file: Name of the source file to refresh
            
        Returns:
            Number of new chunks added
        """
        docs_path = Path(self.docs_directory) / source_file
        
        if not docs_path.exists():
            logger.warning(f"Document not found: {source_file}")
            return 0
        
        # Remove old chunks
        removed = self.remove_documents_by_source(source_file)
        logger.info(f"Removed {removed} old chunks from {source_file}")
        
        # Load and add new version
        try:
            loader = PyPDFLoader(str(docs_path))
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source_file"] = source_file
                doc.metadata["document_type"] = self._categorize_document(source_file)
            
            chunks = self.text_splitter.split_documents(docs)
            
            if self.vectorstore:
                self.vectorstore.add_documents(chunks)
                logger.info(f"Added {len(chunks)} new chunks for {source_file}")
                return len(chunks)
            
            return 0
        except Exception as e:
            logger.error(f"Error refreshing document {source_file}: {e}")
            return 0
    
    def get_index_status(self) -> Dict[str, Any]:
        """Get status of the document index."""
        try:
            if self.vectorstore is None:
                return {
                    "indexed": False,
                    "document_count": 0,
                    "persist_directory": self.persist_directory,
                    "docs_directory": self.docs_directory,
                    "indexed_sources": []
                }
            
            count = self.vectorstore._collection.count()
            indexed_sources = list(self.get_indexed_sources())
            
            # Get PDF files in docs directory
            docs_path = Path(self.docs_directory)
            all_pdfs = [pdf.name for pdf in docs_path.glob("*.pdf")] if docs_path.exists() else []
            unindexed = [pdf for pdf in all_pdfs if pdf not in indexed_sources]
            
            return {
                "indexed": count > 0,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "docs_directory": self.docs_directory,
                "indexed_sources": indexed_sources,
                "unindexed_files": unindexed,
                "total_pdf_files": len(all_pdfs)
            }
        except Exception as e:
            logger.error(f"Error getting index status: {e}")
            return {"indexed": False, "error": str(e)}


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


async def initialize_rag():
    """Initialize RAG service and index documents if needed."""
    service = get_rag_service()
    status = service.get_index_status()
    
    if not status.get("indexed"):
        logger.info("Indexing healthcare documents...")
        count = service.index_documents()
        logger.info(f"Indexed {count} document chunks")
    
    return service
