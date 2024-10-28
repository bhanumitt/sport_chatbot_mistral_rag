import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pickle
import streamlit as st
import gdown
import subprocess

subprocess.run(
        [
            "pip", "install", "llama-cpp-python", 
            "--no-cache-dir", "--force-reinstall", "--verbose", 
            "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/metal/"
        ],
        check=True
    )

from llama_cpp import Llama

class SentenceTransformerRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "embeddings_cache"):
        self.device = torch.device("cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.doc_embeddings = None
        self.cache_dir = cache_dir
        
    def load_specific_cache(self, cache_filename: str) -> dict:
        cache_path = os.path.join(self.cache_dir, cache_filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found at {cache_path}")
        
        print(f"Loading cache from: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def encode(self, texts: list) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return F.normalize(embeddings, p=2, dim=1)

    def store_embeddings(self, embeddings: torch.Tensor):
        self.doc_embeddings = embeddings

    def search(self, query_embedding: torch.Tensor, k: int):
        if self.doc_embeddings is None:
            raise ValueError("No document embeddings stored!")

        similarities = F.cosine_similarity(query_embedding, self.doc_embeddings)
        scores, indices = torch.topk(similarities, k=min(k, similarities.shape[0]))
        
        return indices.cpu(), scores.cpu()

class RAGPipeline:
    def __init__(self, cache_filename: str, k: int = 10):
        self.cache_filename = cache_filename
        self.k = k
        self.retriever = SentenceTransformerRetriever()
        self.documents = []

        # Path to the model
        model_path = "mistral-7b-v0.1.Q4_K_M.gguf"
        
        # Google Drive link
        drive_link = "https://drive.google.com/uc?id=1poGZBUD2vqxp6-9cqCo1X0TQdIZuWFp5"  # replace with your file id/link
        # Check if model exists, otherwise download it
        if not os.path.exists(model_path):
            print("Model not found. Downloading from Google Drive...")
            gdown.download(drive_link, model_path, quiet=False)

        # Load the model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

    def load_cached_embeddings(self):
        """Load embeddings from specific cache file"""
        try:
            cache_data = self.retriever.load_specific_cache(self.cache_filename)
            self.documents = cache_data['documents']
            self.retriever.store_embeddings(cache_data['embeddings'])
            return True
        except Exception as e:
            st.error(f"Error loading cache: {str(e)}")
            return False

    def process_query(self, query: str) -> str:
        MAX_ATTEMPTS = 5
        SIMILARITY_THRESHOLD = 0.4
        
        for attempt in range(MAX_ATTEMPTS):
            try:
                print(f"\nAttempt {attempt + 1}/{MAX_ATTEMPTS}")
                
                # Get query embedding and search for relevant docs
                query_embedding = self.retriever.encode([query])
                indices, _ = self.retriever.search(query_embedding, self.k)
            
                relevant_docs = [self.documents[idx] for idx in indices.tolist()]
                context = "\n".join(relevant_docs)
                
                prompt = f"""Context information is below in backticks:

                ```
                {context}
                ```
                
                Given the context above, please answer the following question:
                {query}

                If you cannot answer it based on the context, please mention politely that you don't know the answer. 
                Prefer to answer whatever information you can give to the user based on the context. 
                Answer in a paragraph format. 
                Answer using the information available in the context.
                Please don't repeat any part of this prompt in the answer. Feel free to use this information to improve the answer.
                Please avoid repetition.  
                
                Answer:"""

                response = self.llm(
                    prompt,
                    max_tokens=1024,
                    temperature=0.4,
                    top_p=0.95,
                    echo=False,
                    stop=["Question:", "\n\n"]
                )

                answer = response['choices'][0]['text'].strip()
                
                # Check if response is empty or too short
                if not answer or len(answer) < 2:
                    print(f"Got empty or too short response: '{answer}'. Retrying...")
                    continue
                
                # Validate response relevance by comparing embeddings
                response_embedding = self.retriever.encode([answer])
                response_similarity = F.cosine_similarity(query_embedding, response_embedding)
                response_score = response_similarity.item()
                print(f"Response relevance score: {response_score:.3f}")
                
                if response_score < SIMILARITY_THRESHOLD:
                    print(f"Response: {answer}. Response relevance {response_score:.3f} below threshold {SIMILARITY_THRESHOLD}. Retrying...")
                    continue
                
                print(f"Successful response generated on attempt {attempt + 1}")
                return answer

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                continue

        return "I apologize, but after multiple attempts, I was unable to generate a satisfactory response. Please try rephrasing your question."


@st.cache_resource
def initialize_rag_pipeline(cache_filename: str):
    """Initialize and load the RAG pipeline with cached embeddings"""
    rag = RAGPipeline(cache_filename)
    success = rag.load_cached_embeddings()
    if not success:
        st.error("Failed to load cached embeddings. Please check the cache file path.")
        st.stop()
    return rag

def main():
    st.title("The Sport Chatbot")
    st.subheader("Using ESPN API")

    st.write("Hey there! 👋 I can help you with information on Ice Hockey, Baseball, American Football, Soccer, and Basketball. With access to the ESPN API, I'm up to date with the latest details for these sports up until October 2024.")
    st.write("Got any general questions? Feel free to ask—I'll do my best to provide answers based on the information I've been trained on!")

    # Use the specific cache file we know exists
    cache_filename = "embeddings_2296.pkl"
    
    try:
        rag = initialize_rag_pipeline(cache_filename)
    except Exception as e:
        st.error(f"Error initializing the application: {str(e)}")
        st.stop()

    # Query input
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching for information..."):
                response = rag.process_query(query)
                st.write("### Answer:")
                st.write(response)
        else:
            st.warning("Please enter a question!")

if __name__ == "__main__":
    main()