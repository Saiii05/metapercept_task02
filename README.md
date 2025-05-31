# RDF Knowledge Graph with Semantic Search

This project demonstrates building a basic RDF knowledge graph, converting its triples to text, embedding these sentences, storing them in FAISS, and performing semantic search using user queries.

## Project Structure

- `knowledge_graph.py`: The main script that builds the knowledge graph, performs embeddings, and answers queries.
- `test_knowledge_graph.py`: Unit tests for the functionalities in `knowledge_graph.py`.
- `requirements.txt`: A file listing necessary Python packages.

## Setup

1.  **Clone the repository (if applicable)**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create and activate a virtual environment (recommended)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    Create a `requirements.txt` file with the following content:
    ```
    rdflib
    sentence-transformers
    faiss-cpu
    numpy
    torch
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

The main application will build a small knowledge graph, process it, and then run a few predefined example queries.

To run the application:
```bash
python knowledge_graph.py
```

Expected output will show:
- The RDF graph in Turtle format.
- The text sentences derived from the graph.
- The generated sentence embeddings (and their shape).
- Confirmation of FAISS index creation.
- Results for predefined queries like "Who is a person?", "What does Bob do?", and "Who works at AcmeCorp?".

## Running Tests

To run the unit tests:
```bash
python test_knowledge_graph.py
```
You should see output indicating that all tests passed.

## How it Works

1.  **Knowledge Graph Creation**: Uses `rdflib` to create an RDF graph with predefined triples (e.g., "Alice is a Person", "Alice knows Bob").
2.  **RDF to Text**: Converts these RDF triples into simple English sentences.
3.  **Sentence Embedding**: Uses `sentence-transformers` (with a model like 'all-MiniLM-L6-v2') to generate vector embeddings for these sentences.
4.  **Embedding Storage**: Stores these embeddings in a FAISS index for efficient similarity search.
5.  **Query Answering**:
    - The current version in `knowledge_graph.py` processes a predefined list of example queries.
    - For each query, it generates an embedding.
    - It then searches the FAISS index for the most similar sentence embeddings.
    - The corresponding text sentences are returned as the answer.
