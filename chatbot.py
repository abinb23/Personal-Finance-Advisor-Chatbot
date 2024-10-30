import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

# Title and Objective
st.markdown("<h1 class='main-title'>Finance Advisor Chatbot 💰🤖</h1>",
            unsafe_allow_html=True)
st.markdown("""
    <p class='objective-text'>
        A Personal finance advisor chatbot capable of assisting users with financial planning, budgeting, and investment strategies. The chatbot uses Retrieval-Augmented Generation (RAG) and a Vector Database to provide personalized financial advice based on relevant data.
    </p>
""", unsafe_allow_html=True)

# Initialize the embeddings model
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use a suitable model
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Sidebar file uploader
st.sidebar.title("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Choose a .csv or .xlsx file", type=["csv", "xlsx"])

# Dummy data for testing (remove this in production)
dummy_data = pd.DataFrame({
    'text': [
        "Invest in mutual funds for long-term gains.",
        "Consider a savings account for emergency funds.",
        "Stocks offer higher returns but come with higher risks.",
    ]
})

# Load data
if uploaded_file:
    # Load the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    # Check if the 'text' column exists
    if 'text' not in df.columns:
        st.warning(
            "The uploaded file must contain a 'text' column. A default 'text' column has been added.")
        df = dummy_data  # Use dummy data if 'text' column is missing
    else:
        # Clean the 'text' column to ensure all entries are strings
        df['text'] = df['text'].fillna('').astype(str)
        st.success("File loaded successfully!")
else:
    st.info("Please upload a .csv or .xlsx file to get started.")
    df = dummy_data  # Use dummy data when no file is uploaded

# Load the documents into FAISS vector store
loader = DataFrameLoader(df, page_content_column='text')
db = FAISS.from_documents(loader.load(), embeddings)

# Initialize session state for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**User:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# Text input box for user to enter questions
user_input = st.text_input("Type your question here:")

# Check if the user has entered any input
if user_input:
    # Perform a similarity search
    results = db.similarity_search(user_input)

    # Create a response based on the search results
    if results:
        response = results[0].page_content
    else:
        response = "I'm sorry, I couldn't find any relevant information."

    # Store chat history in session state
    st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Display updated chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# Ensure the app is responsive
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    st.experimental_rerun()  # Only if still needed, you can remove if this causes issues
