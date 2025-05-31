from rdflib import Graph, Literal, Namespace, RDF, URIRef
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Define global namespaces
EX = Namespace("http://example.org/") # Changed from 'ex' to 'EX' for global const style
# PRED = Namespace("http://example.org/pred/") # Example if more namespaces are needed

# --- Core Functions ---

def create_graph_and_define_elements():
    """Creates the RDF graph and defines initial entities, predicates, and types."""
    g = Graph()
    # Define URIs for entities
    alice = EX.Alice
    bob = EX.Bob
    acme_corp = EX.AcmeCorp

    # Define URIs for predicates (using RDF.type for 'is a')
    knows = EX.knows
    works_at = EX.works_at

    # Define URIs for types
    Person = EX.Person
    Organization = EX.Organization

    # Add triples to the graph
    g.add((alice, RDF.type, Person))
    g.add((bob, RDF.type, Person))
    g.add((alice, knows, bob))
    g.add((bob, works_at, acme_corp))
    g.add((acme_corp, RDF.type, Organization))
    return g, EX # Return graph and the primary namespace

def get_uri_last_part(uri):
    """Extracts the last part of a URI, either fragment or last path segment."""
    if '#' in uri:
        return uri.split('#')[-1]
    return uri.rstrip('/').split('/')[-1]

def rdf_to_text(graph, namespace_map=None):
    """Converts RDF triples to human-readable sentences."""
    sentences = []
    if namespace_map is None:
        namespace_map = {}

    rev_namespace_map = {str(v): k for k, v in namespace_map.items()}

    for s, p, o in graph:
        s_str = str(s)
        if s_str in rev_namespace_map:
            s_display = f"{rev_namespace_map[s_str]}:{get_uri_last_part(s_str)}"
        else:
            s_display = get_uri_last_part(s_str)

        p_str = str(p)
        if p_str == str(RDF.type):
            p_display = "is a"
        elif p_str in rev_namespace_map:
            p_display = f"{rev_namespace_map[p_str]}:{get_uri_last_part(p_str)}"
        else:
            p_display = get_uri_last_part(p_str)
            p_display = ''.join([' ' + char.lower() if char.isupper() else char for char in p_display]).lstrip()
            p_display = p_display.replace('_', ' ')

        if isinstance(o, Literal):
            o_display = o.value
        else:
            o_str = str(o)
            if o_str in rev_namespace_map:
                o_display = f"{rev_namespace_map[o_str]}:{get_uri_last_part(o_str)}"
            else:
                o_display = get_uri_last_part(o_str)

        sentences.append(f"{s_display} {p_display} {o_display}.")
    return sentences

def embed_sentences(sentences, model_name='all-MiniLM-L6-v2'):
    """Generates embeddings for a list of sentences and returns the model."""
    print("\n--- Initializing SBERT Model & Embedding Sentences ---")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    print("Embeddings generated.")
    print("Shape of embeddings:", embeddings.shape)
    return embeddings, model

def create_faiss_index(embeddings):
    """Creates a FAISS index from embeddings."""
    print("\n--- Creating FAISS Index ---")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    # FAISS expects float32
    index.add(embeddings.astype(np.float32))
    print("FAISS index created and embeddings added.")
    print("Number of vectors in index:", index.ntotal)
    return index

def search_in_faiss(query_text, faiss_index, text_sentences, model, k=1):
    """Searches for similar sentences in the FAISS index."""
    query_embedding = model.encode([query_text])
    query_embedding = query_embedding.astype(np.float32)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    distances, indices = faiss_index.search(query_embedding, k)

    retrieved = [text_sentences[i] for i in indices[0]]
    return retrieved

# --- Main Execution ---

def main():
    # 1. Create RDF Graph
    print("--- Creating RDF Graph ---")
    graph, ex_ns = create_graph_and_define_elements() # ex_ns is EX
    print("RDF Graph created.")

    # Print the turtle serialization (optional)
    print("\n--- Turtle Serialization ---")
    print(graph.serialize(format="turtle"))

    # 2. Convert RDF to Text
    print("\n--- Converting RDF to Text ---")
    # Pass ex_ns (which is EX) to the map. RDF is globally available from rdflib.
    text_sentences = rdf_to_text(graph, namespace_map={"ex": ex_ns, "rdf": RDF})
    print("Text sentences extracted:")
    for sentence in text_sentences:
        print(f"- {sentence}")

    # 3. Embed Sentences
    sentence_embeddings, sbert_model = embed_sentences(text_sentences)

    # 4. Create FAISS Index
    faiss_index = create_faiss_index(sentence_embeddings)

    # 5. Simulated Interactive Query Loop (as input() is problematic in this env)
    print("\n--- Simulated Interactive Query ---")
    simulated_queries = [
        "Who is a person?",
        "What does Bob do?",
        "Who works at AcmeCorp?"
    ]

    for user_query in simulated_queries:
        print(f"Simulating query: {user_query}")
        # Retrieve k=2 for potentially more diverse results in simulation
        retrieved = search_in_faiss(user_query, faiss_index, text_sentences, sbert_model, k=2)

        if retrieved:
            print(f"Retrieved: {retrieved}")
        else:
            print("No relevant sentences found.")
        print("-" * 20)

    print("Simulated 'exit'. Exiting.")

if __name__ == "__main__":
    main()
