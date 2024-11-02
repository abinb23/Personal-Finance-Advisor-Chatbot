import streamlit as st
import pandas as pd
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
import os
import time
from datetime import datetime, timedelta

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = None
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# Rate limiting settings
RATE_LIMIT_PERIOD = 60  # seconds
MAX_QUERIES_PER_PERIOD = 10
QUERY_COOLDOWN = 2  # seconds between individual queries


class RateLimiter:
    @staticmethod
    def can_make_request():
        current_time = datetime.now()

        # Check cooldown between individual queries
        if st.session_state.last_query_time:
            time_since_last_query = (
                current_time - st.session_state.last_query_time).total_seconds()
            if time_since_last_query < QUERY_COOLDOWN:
                return False, f"Please wait {QUERY_COOLDOWN - time_since_last_query:.1f} seconds before sending another query"

        # Reset counter if period has elapsed
        if st.session_state.last_query_time:
            if (current_time - st.session_state.last_query_time).total_seconds() > RATE_LIMIT_PERIOD:
                st.session_state.query_count = 0

        # Check rate limit
        if st.session_state.query_count >= MAX_QUERIES_PER_PERIOD:
            time_until_reset = RATE_LIMIT_PERIOD - \
                (current_time - st.session_state.last_query_time).total_seconds()
            return False, f"Rate limit exceeded. Please wait {time_until_reset:.0f} seconds before trying again"

        return True, ""


# Title and Objective
st.markdown("<h1 class='main-title'>Finance Advisor Chatbot 💰🤖</h1>",
            unsafe_allow_html=True)
st.markdown("""
    <p class='objective-text'>
        A Personal finance advisor chatbot capable of assisting users with financial planning, budgeting, and investment strategies. The chatbot uses Retrieval-Augmented Generation (RAG) and a Vector Database to provide personalized financial advice based on relevant data.
    </p>
""", unsafe_allow_html=True)

# Load and prepare data


@st.cache_resource
def initialize_database():
    csv_file_path = r"C:\Users\ASUS\Documents\Abin\talrop\gen-ai\task\Financial_Info.csv"

    if not os.path.isfile(csv_file_path):
        st.error(
            "The specified CSV file does not exist or is not accessible. Please check the file path.")
        st.stop()

    df = pd.read_csv(csv_file_path)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    page_content_columns = ['Investment_Option', 'Financial_Product', 'Budgeting_Strategy',
                            'Retirement_Plan', 'Risk_Level', 'Interest_Rate', 'Annual_Return']

    df['combined_text'] = df[page_content_columns].astype(
        str).agg(' '.join, axis=1)

    try:
        loader = DataFrameLoader(df, page_content_column='combined_text')
        return FAISS.from_documents(loader.load(), embeddings)
    except KeyError as e:
        st.error(f"KeyError: {str(e)}. Please check the column names.")
        st.stop()


db = initialize_database()

# Display chat history
st.markdown("### Chat History")
for chat in st.session_state.chat_history:
    with st.container():
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")

# Text input box with clear button
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Type your question here:", key="user_input")
with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.rerun()

# Process user input
if user_input:
    # Check rate limiting
    can_request, message = RateLimiter.can_make_request()

    if not can_request:
        st.warning(message)
    else:
        with st.spinner('Generating response...'):
            # Update rate limiting counters
            st.session_state.last_query_time = datetime.now()
            st.session_state.query_count += 1

            # Perform similarity search
            results = db.similarity_search(user_input)

            if results:
                context = results[0].page_content

                try:
                    response = requests.post(
                        "https://api-inference.huggingface.co/models/gpt2",
                        headers={
                            "Authorization": "Bearer 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"},
                        json={
                            "inputs": f"{context}\n\nUser: {user_input}",
                            "parameters": {"max_length": 100}
                        }
                    )

                    if response.status_code == 200:
                        response_text = response.json(
                        )[0]["generated_text"].strip()
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "bot": response_text
                        })
                        st.rerun()
                    else:
                        st.error(f"API Error: {
                                 response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.info(
                    "I couldn't find any relevant information for your question. Please try rephrasing it.")

# Display query limit status
remaining_queries = MAX_QUERIES_PER_PERIOD - st.session_state.query_count
if remaining_queries < MAX_QUERIES_PER_PERIOD:
    st.sidebar.info(f"Remaining queries in current period: {
                    remaining_queries}")
