import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import chromadb
import re
import streamlit as st
from typing import List, Dict
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up Azure OpenAI credentials
os.environ["OPENAI_API_KEY"] = "BPcV5t4PYdBfLcbEKeOItVL3UgmyOI3pvoACKhwnrB5iVUj13CchJQQJ99BGACYeBjFXJ3w3AAABACOGraLO"
os.environ["OPENAI_ENDPOINT"] = "https://genaideployment.openai.azure.com"
os.environ["OPENAI_DEPLOYMENT"] = "gpt-4o"
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_endpoint=os.environ["OPENAI_ENDPOINT"]
    )
    azure_client_available = True
except:
    st.warning("Azure OpenAI client could not be initialized. Using fallback mode.")
    azure_client_available = False

# Initialize ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_available = True
except Exception as e:
    st.error(f"ChromaDB initialization failed: {e}")
    chroma_available = False

# Create or get collection
collection_name = "medical_faqs"
if chroma_available:
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name)
else:
    collection = None

class MedicalFAQChatbot:
    def __init__(self):
        self.collection = collection
        self.llm_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.data_loaded = False
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.fallback_mode = not chroma_available
        
        # Try to load sentence-transformers, but have fallback ready
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_available = True
        except ImportError:
            st.warning("sentence-transformers not available. Using TF-IDF fallback for embeddings.")
            self.embedding_available = False
            self.fallback_mode = True
        except Exception as e:
            st.warning(f"Could not load embedding model: {e}. Using TF-IDF fallback.")
            self.embedding_available = False
            self.fallback_mode = True
        
        # Greeting patterns
        self.greeting_patterns = [
            r"hello", r"hi", r"hey", r"greetings", r"good morning", 
            r"good afternoon", r"good evening", r"howdy", r"what's up",
            r"hola", r"namaste", r"yo"
        ]
        
        # Thank you patterns
        self.thanks_patterns = [
            r"thank", r"thanks", r"appreciate", r"grateful", r"helpful"
        ]
        
        # Farewell patterns
        self.farewell_patterns = [
            r"bye", r"goodbye", r"see you", r"farewell", r"take care"
        ]
    
    def is_greeting(self, text):
        """Check if the text is a greeting"""
        text = text.lower()
        for pattern in self.greeting_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def is_thanks(self, text):
        """Check if the text is expressing thanks"""
        text = text.lower()
        for pattern in self.thanks_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def is_farewell(self, text):
        """Check if the text is a farewell"""
        text = text.lower()
        for pattern in self.farewell_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def generate_greeting(self):
        """Generate a friendly greeting"""
        greetings = [
            "Hello! I'm your medical assistant. How can I help you today?",
            "Hi there! I'm here to answer your medical questions.",
            "Greetings! I'm a medical FAQ assistant. What would you like to know?",
            "Hello! I can help answer your medical questions based on our knowledge base."
        ]
        return np.random.choice(greetings)
    
    def generate_thanks_response(self):
        """Generate a response to thanks"""
        responses = [
            "You're welcome! I'm glad I could help.",
            "Happy to assist! Let me know if you have any other questions.",
            "You're very welcome! Feel free to ask if you need more information.",
            "My pleasure! Don't hesitate to reach out if you have more questions."
        ]
        return np.random.choice(responses)
    
    def generate_farewell(self):
        """Generate a farewell message"""
        farewells = [
            "Goodbye! Take care and stay healthy!",
            "See you later! Feel free to come back with any medical questions.",
            "Farewell! Remember to consult a healthcare professional for personal medical advice.",
            "Take care! I'm here if you have more questions later."
        ]
        return np.random.choice(farewells)
    
    def load_and_preprocess_data(self, file_path: str):
        """Load and preprocess the medical FAQ dataset from the given path"""
        try:
            # Load the dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide CSV or JSON.")
            
            # Display dataset info
            st.info(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
            st.write("Columns found:", list(df.columns))
            
            # Basic data cleaning
            df = df.dropna()  # Remove rows with missing values
            df = df.drop_duplicates()  # Remove duplicates
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Rename columns to expected format if needed
            column_mapping = {}
            for col in df.columns:
                if 'question' in col:
                    column_mapping[col] = 'question'
                elif 'answer' in col:
                    column_mapping[col] = 'answer'
                elif 'categor' in col:  # category or categories
                    column_mapping[col] = 'category'
                elif 'keyword' in col:
                    column_mapping[col] = 'keywords'
            
            df = df.rename(columns=column_mapping)
            
            # Ensure we have at least question and answer columns
            if 'question' not in df.columns or 'answer' not in df.columns:
                st.error("Dataset must contain 'question' and 'answer' columns")
                return None
            
            # Fill missing values for optional columns
            if 'category' not in df.columns:
                df['category'] = 'General'
            if 'keywords' not in df.columns:
                df['keywords'] = ''
                
            self.df = df
            self.data_loaded = True
            
            # Prepare TF-IDF as fallback
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            combined_texts = [f"{q} {a}" for q, a in zip(df['question'], df['answer'])]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_texts)
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def generate_embeddings(self, texts: List[str]):
        """Generate embeddings using a local model or fallback"""
        try:
            if self.embedding_available:
                # Generate embeddings using local model
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            else:
                # Fallback to TF-IDF
                tfidf_vectors = self.tfidf_vectorizer.transform(texts)
                return tfidf_vectors.toarray().tolist()
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def clean_text(self, text: str):
        """Clean text by removing special characters and extra spaces"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.strip()
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            if not self.fallback_mode:
                # Get all IDs first
                results = self.collection.get()
                if results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    st.info("Cleared existing data from collection")
        except Exception as e:
            st.warning(f"Could not clear collection: {str(e)}")
    
    def index_faqs(self, df):
        """Index FAQs in the vector database or prepare fallback"""
        try:
            if not self.fallback_mode:
                # Clear existing data
                self.clear_collection()
                
                # Create combined text for embedding
                texts = []
                metadatas = []
                ids = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    # Create a combined text representation
                    combined_text = f"Question: {row.get('question', '')} Answer: {row.get('answer', '')}"
                    if 'category' in row:
                        combined_text += f" Category: {row['category']}"
                    if 'keywords' in row:
                        combined_text += f" Keywords: {row['keywords']}"
                    
                    texts.append(combined_text)
                    
                    # Create metadata
                    metadata = {
                        'question': row.get('question', ''),
                        'answer': row.get('answer', ''),
                        'category': row.get('category', ''),
                        'keywords': row.get('keywords', ''),
                        'original_index': idx
                    }
                    metadatas.append(metadata)
                    ids.append(str(idx))
                    
                    # Update progress
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {idx + 1}/{len(df)} FAQs...")
                
                # Generate embeddings in batches to avoid memory issues
                batch_size = 100
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.generate_embeddings(batch_texts)
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
                        st.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    time.sleep(0.1)  # Small delay to prevent UI freeze
                
                if all_embeddings and len(all_embeddings) == len(texts):
                    # Add to collection
                    self.collection.add(
                        embeddings=all_embeddings,
                        metadatas=metadatas,
                        ids=ids,
                        documents=texts
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Successfully indexed {len(ids)} FAQs!")
                    return True
                else:
                    st.error("Failed to generate embeddings for all texts")
                    return False
            else:
                # Fallback mode - just prepare the data
                st.info("Using fallback mode (TF-IDF similarity)")
                self.df = df
                self.data_loaded = True
                return True
                
        except Exception as e:
            st.error(f"Error indexing FAQs: {str(e)}")
            self.fallback_mode = True
            return True  # Continue with fallback mode
    
    def retrieve_relevant_faqs(self, query: str, k: int = 5):
        """Retrieve relevant FAQs based on the query"""
        try:
            if not self.fallback_mode and self.collection is not None:
                # Generate embedding for the query
                query_embedding = self.generate_embeddings([query])
                if not query_embedding:
                    return self.fallback_retrieval(query, k)
                    
                query_embedding = query_embedding[0]
                
                # Query the collection
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=["metadatas", "documents", "distances"]
                )
                
                return results
            else:
                # Use fallback retrieval
                return self.fallback_retrieval(query, k)
        except Exception as e:
            st.error(f"Error retrieving FAQs: {str(e)}")
            return self.fallback_retrieval(query, k)
    
    def fallback_retrieval(self, query: str, k: int = 5):
        """Fallback retrieval using TF-IDF similarity"""
        try:
            if self.df is None or self.tfidf_vectorizer is None:
                return None
                
            # Transform query to TF-IDF
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-k:][::-1]
            
            # Prepare results in Chroma-like format
            results = {
                'ids': [[str(i) for i in top_indices]],
                'metadatas': [[{
                    'question': self.df.iloc[i]['question'],
                    'answer': self.df.iloc[i]['answer'],
                    'category': self.df.iloc[i].get('category', ''),
                    'keywords': self.df.iloc[i].get('keywords', ''),
                    'original_index': i
                } for i in top_indices]],
                'distances': [[1 - similarities[i] for i in top_indices]],
                'documents': [[f"Question: {self.df.iloc[i]['question']} Answer: {self.df.iloc[i]['answer']}" for i in top_indices]]
            }
            
            return results
        except Exception as e:
            st.error(f"Error in fallback retrieval: {str(e)}")
            return None
    
    def generate_response(self, query: str, context: List[Dict]):
        """Generate a response using Azure OpenAI GPT model or fallback"""
        try:
            if not azure_client_available:
                return self.fallback_response(query, context)
                
            # Prepare context for the prompt
            context_text = ""
            for i, item in enumerate(context):
                context_text += f"FAQ {i+1}:\nQuestion: {item['question']}\nAnswer: {item['answer']}\n"
                if item.get('category'):
                    context_text += f"Category: {item['category']}\n"
                if item.get('keywords'):
                    context_text += f"Keywords: {item['keywords']}\n"
                context_text += "\n"
            
            # Create the prompt with instructions for better interaction
            prompt = f"""
            You are a friendly medical assistant chatbot. Your role is to provide accurate, helpful information based on the context provided while being engaging and conversational.

            CONTEXT INFORMATION (medical FAQs from our database):
            {context_text}

            USER QUESTION: {query}

            IMPORTANT INSTRUCTIONS:
            1. Answer the question based PRIMARILY on the context provided above.
            2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question accurately. Please consult a healthcare professional for specific medical advice."
            3. Be conversational, friendly, and empathetic in your responses.
            4. If the question is not related to medical topics, politely redirect to medical questions.
            5. NEVER make up information or provide medical advice beyond what's in the context.
            6. If multiple FAQs are relevant, synthesize the information into a coherent, summarized response.
            7. Add a brief summary at the end if the answer is complex.
            8. Use a warm, professional tone suitable for healthcare communication.
            9. If appropriate, ask follow-up questions to better understand the user's needs.

            RESPONSE:
            """
            
            # Generate response
            response = client.chat.completions.create(
                model=self.llm_deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly medical assistant that provides accurate information based on the provided context while maintaining a warm, professional tone."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly higher temperature for more varied responses
                max_tokens=600
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return self.fallback_response(query, context)
    
    def fallback_response(self, query: str, context: List[Dict]):
        """Fallback response when Azure OpenAI is not available"""
        if not context:
            return "I don't have enough information to answer that question accurately. Please consult a healthcare professional for specific medical advice."
        
        # Simple concatenation of the most relevant answers with summary
        answers = [item['answer'] for item in context[:2]]  # Use top 2 answers
        response = "Based on our medical knowledge base:\n\n"
        
        for i, answer in enumerate(answers):
            response += f"{answer}\n\n"
        
        # Add a simple summary
        response += "In summary, this information is based on our medical FAQ database. For personalized advice, please consult a healthcare professional."
        
        return response
    
    def process_query(self, query: str):
        """Process a user query and return a response"""
        # Handle greetings
        if self.is_greeting(query):
            return self.generate_greeting(), []
        
        # Handle thanks
        if self.is_thanks(query):
            return self.generate_thanks_response(), []
        
        # Handle farewells
        if self.is_farewell(query):
            return self.generate_farewell(), []
        
        # Check if dataset is loaded
        if not self.data_loaded:
            return "Please load and index the dataset first using the sidebar button.", []
            
        # Retrieve relevant FAQs
        results = self.retrieve_relevant_faqs(query)
        
        if not results or not results['metadatas']:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your question or ask about a different medical topic.", []
        
        # Extract metadata from results
        context = []
        for metadata_list in results['metadatas']:
            for metadata in metadata_list:
                context.append(metadata)
        
        # Generate response
        response = self.generate_response(query, context)
        
        return response, context

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical FAQ Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Medical FAQ Chatbot")
    st.markdown("Ask questions about medical topics and get accurate, friendly answers from our FAQ database.")
    
    # Initialize chatbot
    chatbot = MedicalFAQChatbot()
    
    # Your dataset path
    dataset_path = r"C:\Users\prasad\OneDrive\Desktop\medical-rag-chatbot\data\sample_faqs.csv"
    
    # Sidebar for data management
    with st.sidebar:
        st.header("Data Management")
        
        if st.button("Load and Index Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                df = chatbot.load_and_preprocess_data(dataset_path)
                
                if df is not None:
                    st.dataframe(df.head(3))
                    st.write(f"Dataset shape: {df.shape}")
                    
                    with st.spinner("Indexing FAQs..."):
                        success = chatbot.index_faqs(df)
                        if success:
                            st.success("FAQs indexed successfully!")
                            st.session_state.dataset_loaded = True
                            st.rerun()
        
        st.divider()
        st.header("System Status")
        st.write(f"OpenAI: {'✅ Available' if azure_client_available else '❌ Not Available'}")
        st.write(f"ChromaDB: {'✅ Available' if chroma_available else '❌ Not Available'}")
        st.write(f"Embedding Model: {'✅ Available' if chatbot.embedding_available else '❌ Using TF-IDF Fallback'}")
        st.write(f"Current Mode: {'🔧 Fallback' if chatbot.fallback_mode else '🚀 Full RAG'}")
        st.write(f"Dataset Loaded: {'✅ Yes' if chatbot.data_loaded else '❌ No'}")
        
        if st.button("Clear Chat History"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
            
        st.divider()
        st.header("How to Use")
        st.markdown("""
        1. Click 'Load and Index Dataset' to load your medical FAQs
        2. Start chatting by typing medical questions
        3. The chatbot will provide answers based on the FAQ database
        4. View sources by expanding the 'View sources' section
        """)
    
    # Check if dataset is loaded
    if hasattr(st.session_state, 'dataset_loaded') and st.session_state.dataset_loaded:
        chatbot.data_loaded = True
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your medical FAQ assistant. Please load the dataset using the sidebar to get started."}]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question or say hello..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, context = chatbot.process_query(prompt)
                
                # Display response
                st.markdown(response)
                
                # Optionally display context
                if context and chatbot.data_loaded:
                    with st.expander("View sources used for this answer"):
                        for i, item in enumerate(context):
                            st.markdown(f"**FAQ {i+1}:**")
                            st.markdown(f"**Question:** {item.get('question', 'N/A')}")
                            st.markdown(f"**Answer:** {item.get('answer', 'N/A')}")
                            if item.get('category'):
                                st.markdown(f"**Category:** {item.get('category')}")
                            if item.get('keywords'):
                                st.markdown(f"**Keywords:** {item.get('keywords')}")
                            st.markdown("---")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()