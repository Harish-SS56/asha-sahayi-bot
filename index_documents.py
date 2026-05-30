"""
Script to initialize and update the RAG index from healthcare documents.
Run this to index new documents added to the docs/ folder.

Usage:
  python index_documents.py          # Add new documents only (incremental)
  python index_documents.py --status # Show index status
  python index_documents.py --full   # Full reindex (delete and rebuild)
  python index_documents.py --refresh <filename>  # Re-index specific file
"""

import sys
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def show_status(rag_service):
    """Display current index status."""
    status = rag_service.get_index_status()
    
    print("\n" + "=" * 60)
    print("📚 RAG Index Status")
    print("=" * 60)
    print(f"Indexed: {'✅ Yes' if status.get('indexed') else '❌ No'}")
    print(f"Total chunks in index: {status.get('document_count', 0)}")
    print(f"Persist directory: {status.get('persist_directory')}")
    print(f"Docs directory: {status.get('docs_directory')}")
    print(f"Total PDF files: {status.get('total_pdf_files', 0)}")
    
    indexed_sources = status.get('indexed_sources', [])
    if indexed_sources:
        print(f"\n📄 Indexed Documents ({len(indexed_sources)}):")
        for src in sorted(indexed_sources):
            print(f"  ✓ {src}")
    
    unindexed = status.get('unindexed_files', [])
    if unindexed:
        print(f"\n📋 Unindexed Documents ({len(unindexed)}):")
        for src in sorted(unindexed):
            print(f"  • {src}")
    
    print("=" * 60 + "\n")
    return status


def add_new_documents(rag_service):
    """Add only new documents (incremental indexing)."""
    logger.info("Checking for new documents to index...")
    
    result = rag_service.add_new_documents()
    
    print("\n" + "=" * 60)
    print("📥 Incremental Indexing Result")
    print("=" * 60)
    
    if result['new_files']:
        print(f"✅ Added {result['added']} chunks from {len(result['new_files'])} new file(s):")
        for f in result['new_files']:
            print(f"  + {f}")
    else:
        print("ℹ️  No new documents to add.")
    
    if result['skipped_files']:
        print(f"\n⏭️  Skipped {result['skipped']} already-indexed file(s):")
        for f in result['skipped_files'][:5]:  # Show first 5
            print(f"  • {f}")
        if len(result['skipped_files']) > 5:
            print(f"  ... and {len(result['skipped_files']) - 5} more")
    
    print("=" * 60 + "\n")
    return result


def full_reindex(rag_service):
    """Delete and rebuild the entire index."""
    status = rag_service.get_index_status()
    
    if status.get("indexed"):
        response = input(f"⚠️  Index has {status['document_count']} chunks. Delete and rebuild? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return None
    
    logger.info("Starting full reindex...")
    count = rag_service.index_documents(force_reindex=True)
    
    print("\n" + "=" * 60)
    print("🔄 Full Reindex Complete")
    print("=" * 60)
    print(f"✅ Indexed {count} document chunks")
    print("=" * 60 + "\n")
    return count


def refresh_document(rag_service, filename):
    """Re-index a specific document."""
    logger.info(f"Refreshing document: {filename}")
    count = rag_service.refresh_document(filename)
    
    print("\n" + "=" * 60)
    print(f"🔄 Document Refresh: {filename}")
    print("=" * 60)
    if count > 0:
        print(f"✅ Re-indexed {count} chunks")
    else:
        print("❌ No chunks indexed (file not found or error)")
    print("=" * 60 + "\n")
    return count


def test_retrieval(rag_service):
    """Test retrieval with sample queries."""
    import asyncio
    
    test_queries = [
        "What are the symptoms of hypertension?",
        "How to care for a newborn baby?",
        "What are the danger signs in pregnancy?"
    ]
    
    print("\n🔍 Testing Retrieval...")
    for query in test_queries:
        results = asyncio.run(rag_service.retrieve_relevant_context(query, k=2))
        if results:
            print(f"  ✓ '{query[:40]}...' → {len(results)} results")
        else:
            print(f"  ✗ '{query[:40]}...' → No results")


def main():
    parser = argparse.ArgumentParser(
        description="Manage RAG document index for ASHA AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python index_documents.py              # Add new documents only
  python index_documents.py --status     # Show current index status
  python index_documents.py --full       # Full reindex (rebuild from scratch)
  python index_documents.py --refresh guidelines.pdf  # Re-index specific file
        """
    )
    parser.add_argument('--status', action='store_true', help='Show index status only')
    parser.add_argument('--full', action='store_true', help='Full reindex (delete and rebuild)')
    parser.add_argument('--refresh', type=str, metavar='FILE', help='Re-index a specific file')
    parser.add_argument('--test', action='store_true', help='Test retrieval after indexing')
    
    args = parser.parse_args()
    
    try:
        rag_service = get_rag_service()
        
        if args.status:
            show_status(rag_service)
        elif args.full:
            full_reindex(rag_service)
            if args.test:
                test_retrieval(rag_service)
        elif args.refresh:
            refresh_document(rag_service, args.refresh)
            if args.test:
                test_retrieval(rag_service)
        else:
            # Default: incremental add
            show_status(rag_service)
            add_new_documents(rag_service)
            if args.test:
                test_retrieval(rag_service)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
