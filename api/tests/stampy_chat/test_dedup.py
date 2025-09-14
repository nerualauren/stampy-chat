#!/usr/bin/env python3
"""
Quick test to verify the deduplication logic in retrieve_docs
"""

from unittest.mock import Mock, patch
from stampy_chat.citations import retrieve_docs
from stampy_chat.settings import Settings


def test_deduplication():
    """Test that only highest-scoring chunk per document is returned"""

    # Mock Pinecone client and results
    with patch("stampy_chat.citations.Pinecone") as mock_pinecone_class:
        with patch("stampy_chat.citations.embed_query") as mock_embed:

            mock_embed.return_value = [0.1, 0.2, 0.3]

            # Create mock matches with duplicates
            mock_match1 = Mock()
            mock_match1.score = 0.9
            mock_match1.metadata = {
                "title": "Document A",
                "url": "https://example.com/doc-a#section1",
                "text": "First chunk from doc A",
                "authors": ["Author 1"]
            }

            mock_match2 = Mock()  # Same doc, lower score
            mock_match2.score = 0.7
            mock_match2.metadata = {
                "title": "Document A",
                "url": "https://example.com/doc-a#section2",
                "text": "Second chunk from doc A",
                "authors": ["Author 1"]
            }

            mock_match3 = Mock()  # Different doc
            mock_match3.score = 0.8
            mock_match3.metadata = {
                "title": "Document B",
                "url": "https://example.com/doc-b",
                "text": "Chunk from doc B",
                "authors": ["Author 2"]
            }

            mock_match4 = Mock()  # Same URL as doc A but no title - should be deduplicated
            mock_match4.score = 0.95
            mock_match4.metadata = {
                "title": "",  # Empty title
                "url": "https://example.com/doc-a",  # Same base URL
                "text": "Another chunk from doc A",
                "authors": ["Author 1"]
            }

            # Mock the query result
            mock_results = Mock()
            mock_results.matches = [mock_match1, mock_match2, mock_match3, mock_match4]

            # Mock the index
            mock_index = Mock()
            mock_index.query_namespaces.return_value = mock_results

            # Mock the Pinecone client
            mock_pc = Mock()
            mock_pc.Index.return_value = mock_index
            mock_pinecone_class.return_value = mock_pc

            # Test
            settings = Settings()
            results = retrieve_docs("test query", settings)

            print(f"Number of results: {len(results)}")

            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Title: {result['title']}")
                print(f"  URL: {result['url']}")
                print(f"  Text: {result['text'][:50]}...")
                print(f"  Reference: {result['reference']}")
                print()

            # Should only have 2 results: one from doc A (highest scoring) and one from doc B
            assert len(results) == 2, f"Expected 2 results, got {len(results)}"

            # Results should be sorted by score (descending)
            # The empty title match4 (score 0.95) should win over match1 (score 0.9) for doc A
            # match3 (score 0.8) should be included for doc B

            titles = [r['title'] for r in results]
            texts = [r['text'] for r in results]

            print(f"Result titles: {titles}")
            print(f"Result texts: {texts}")

            # The highest scoring chunk from doc A should be included
            assert any("Another chunk from doc A" in text for text in texts), "Should include highest-scoring chunk from doc A"
            assert any("Chunk from doc B" in text for text in texts), "Should include chunk from doc B"

            # Should NOT include the lower-scoring chunks from doc A
            assert not any("First chunk from doc A" in text for text in texts), "Should not include lower-scoring chunk from doc A"
            assert not any("Second chunk from doc A" in text for text in texts), "Should not include lower-scoring chunk from doc A"
