import unittest
import numpy as np
from rdflib import Namespace, RDF # URIRef might not be directly needed in tests if EX namespace is used

# Import functions from knowledge_graph
from knowledge_graph import (
    create_graph_and_define_elements,
    rdf_to_text,
    embed_sentences,
    create_faiss_index,
    search_in_faiss,
    EX # Import the global namespace
)

class TestKnowledgeGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests in this class."""
        print("Setting up TestKnowledgeGraph resources...")
        cls.graph, cls.ex_ns = create_graph_and_define_elements()

        # Ensure EX from knowledge_graph.py is used if needed for namespace_map
        # or ensure cls.ex_ns is correctly used.
        cls.text_sentences = rdf_to_text(cls.graph, namespace_map={"ex": cls.ex_ns, "rdf": RDF})

        # Ensure embeddings are float32 for FAISS
        raw_embeddings, cls.sbert_model = embed_sentences(cls.text_sentences)
        cls.embeddings = raw_embeddings.astype(np.float32)

        cls.faiss_index = create_faiss_index(cls.embeddings)
        print("TestKnowledgeGraph resources set up.")

    def test_graph_creation(self):
        """Test if the graph is created and has some triples."""
        self.assertTrue(len(self.graph) > 0, "Graph should not be empty.")
        # Example: Check for a specific triple (optional, more detailed)
        # alice = self.ex_ns.Alice
        # person = self.ex_ns.Person
        # self.assertIn((alice, RDF.type, person), self.graph, "Alice should be a Person")

    def test_rdf_to_text_conversion(self):
        """Test the conversion of RDF triples to text sentences."""
        expected_sentences = [
            "Alice is a Person.",
            "Bob is a Person.",
            "Alice knows Bob.",
            "Bob works at AcmeCorp.",
            "AcmeCorp is a Organization."
        ]
        # Use assertCountEqual because the order of sentences from the graph is not guaranteed
        self.assertCountEqual(self.text_sentences, expected_sentences,
                              f"Text sentences do not match expected. Got: {self.text_sentences}")

    def test_embeddings_generation(self):
        """Test if embeddings are generated correctly."""
        self.assertIsNotNone(self.embeddings, "Embeddings should not be None.")
        self.assertEqual(self.embeddings.shape[0], len(self.text_sentences), "Number of embeddings should match number of sentences.")
        self.assertEqual(self.embeddings.shape[1], 384, "Embedding dimension should be 384 for all-MiniLM-L6-v2.")
        self.assertEqual(self.embeddings.dtype, np.float32, "Embeddings should be float32 for FAISS.")

    def test_faiss_index_creation(self):
        """Test if the FAISS index is created correctly."""
        self.assertIsNotNone(self.faiss_index, "FAISS index should not be None.")
        self.assertEqual(self.faiss_index.ntotal, len(self.text_sentences), "Number of vectors in FAISS index should match number of sentences.")

    def test_query_alice_knows(self):
        """Test a query about who Alice knows."""
        query = "Who does Alice know?"
        results = search_in_faiss(query, self.faiss_index, self.text_sentences, self.sbert_model, k=1)
        self.assertIn("Alice knows Bob.", results, f"Query '{query}' failed. Got: {results}")

    def test_query_bob_works_at(self):
        """Test a query about where Bob works."""
        query = "Where does Bob work?" # Or "What organization does Bob work for?"
        # Increase k to 2 or 3 to see if the desired result is nearby
        results = search_in_faiss(query, self.faiss_index, self.text_sentences, self.sbert_model, k=2)
        self.assertIn("Bob works at AcmeCorp.", results, f"Query '{query}' failed to find 'Bob works at AcmeCorp.' within top {len(results)} results. Got: {results}")

    def test_query_who_is_person(self):
        """Test a query asking who is a person, expecting multiple results."""
        query = "Who is a person?"
        results = search_in_faiss(query, self.faiss_index, self.text_sentences, self.sbert_model, k=2)
        self.assertIn("Alice is a Person.", results, f"Query '{query}' failed to find Alice. Got: {results}")
        self.assertIn("Bob is a Person.", results, f"Query '{query}' failed to find Bob. Got: {results}")
        self.assertEqual(len(results), 2, f"Query '{query}' should return 2 results. Got: {results}")

if __name__ == '__main__':
    unittest.main()
