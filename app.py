import streamlit as st
import asyncio
import datetime
from dataclasses import dataclass
from typing import Any, List
from pydantic import BaseModel, Field
from tavily import TavilyClient
from pydantic_ai import Agent, RunContext

# Page config
st.set_page_config(
    page_title="Web AI Researcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .search-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .results-container {
        margin-top: 2rem;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .element-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .query-log {
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f2f6;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    .big-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .category-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .example-button {
        margin: 5px 0;
        border: none;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    .example-button:hover {
        background-color: #e9ecef;
        transform: translateX(5px);
    }
    </style>
""", unsafe_allow_html=True)

# Models and Classes setup
@dataclass
class SearchDataclass:
    max_results: int
    todays_date: str
    search_engines: List[str] = None

class WebSearchResult(BaseModel):
    web_search_title: str = Field(
        description="This is a top level Markdown heading that covers the topic of the query and answer."
    )
    web_search_main: str = Field(
        description="This is a main section that provides answers for the query and web search."
    )
    web_search_bullets: str = Field(
        description="This is a set of bulletpoints that summarize the answers for the query."
    )
    visited_urls: List[str] = Field(
        description="This is a list of URLs that were visited to get the information."
    )

# Initialize session states
if 'tavily_api_key' not in st.session_state:
    st.session_state.tavily_api_key = ''
if 'search_queries' not in st.session_state:
    st.session_state.search_queries = []

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key input with styling
    st.markdown("### 🔑 API Key")
    api_key = st.text_input(
        "Enter your Tavily API key:",
        type="password",
        value=st.session_state.tavily_api_key,
        help="Get your API key from https://tavily.com"
    )
    
    if api_key != st.session_state.tavily_api_key:
        st.session_state.tavily_api_key = api_key
    
    # Settings
    st.markdown("### 🎯 Search Settings")
    max_results = st.slider(
        "Results per search",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of results to fetch per search query"
    )

# Initialize client if API key is provided
tavily_client = None
if st.session_state.tavily_api_key:
    tavily_client = TavilyClient(api_key=st.session_state.tavily_api_key)

# Create the agent
search_agent = Agent(
    "openai:gpt-4o",
    deps_type=SearchDataclass,
    result_type=WebSearchResult,
    system_prompt=(
        "You are a helpful research assistant and expert in research. "
        "When given a question, generate strong keywords to perform 3-5 searches "
        "and combine the results into a comprehensive, well-structured response."
    ),
)

@search_agent.tool
async def get_search(search_data: RunContext[SearchDataclass], query: str, query_number: int) -> dict[str, Any]:
    """Get the search results for a keyword query."""
    if query not in st.session_state.search_queries:
        st.session_state.search_queries.append(query)
    
    with st.empty():
        st.markdown(f"<div class='query-log'>🔍 Search Query {len(st.session_state.search_queries)}: **{query}**</div>", unsafe_allow_html=True)
        max_results = search_data.deps.max_results
        results = tavily_client.get_search_context(query=query, max_results=max_results)
        return results

async def run_search(query: str, max_results: int = 3):
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    deps = SearchDataclass(max_results=max_results, todays_date=current_date)
    
    # Reset search queries
    st.session_state.search_queries = []
    
    try:
        result = await search_agent.run(query, deps=deps)
        return result.data
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Main content
st.markdown("<div class='big-title'>🔬 Web AI Researcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by AI to deliver comprehensive research results</div>", unsafe_allow_html=True)

if not st.session_state.tavily_api_key:
    st.warning("👈 Please enter your Tavily API key in the sidebar to continue.")

# Search interface
with st.container():
    st.markdown("### 🔍 Research Query")
    with st.container():
        query = st.text_area(
            "",
            placeholder="Enter your research question here...",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            search_button = st.button("🔍 Start Research", type="primary", use_container_width=True)

# Example questions in a more organized layout
st.markdown("### 📚 Example Research Topics")

example_questions = [
    {
        "category": "🧬 Science & Technology",
        "questions": [
            "What are the latest breakthroughs in quantum computing and their potential applications?",
            "How is CRISPR gene editing technology being used in medicine today?",
            "What are the most promising developments in fusion energy research?",
        ]
    },
    {
        "category": "💡 Business & Innovation",
        "questions": [
            "How are companies using artificial intelligence to combat climate change?",
            "What are the most successful space tourism ventures and their future plans?",
            "How has remote work transformed company culture and productivity?",
        ]
    }
]

cols = st.columns(2)
for i, category in enumerate(example_questions):
    with cols[i]:
        with st.container():
            st.markdown(f"#### {category['category']}")
            for question in category['questions']:
                if st.button(f"🔍 {question}", key=question, use_container_width=True):
                    query = question
                    search_button = True

# Process search
if search_button and query and tavily_client:
    with st.spinner("Researching your query..."):
        # Show queries in real-time
        st.markdown("### 🔄 Search Queries")
        queries_placeholder = st.empty()
        
        result = asyncio.run(run_search(query, max_results))
        
        if result:
            # Results container
            st.markdown("### 📊 Research Results")
            with st.container():
                # Title
                st.markdown(f"# {result.web_search_title}")
                
                # Main content
                st.markdown("## Detailed Information")
                st.markdown(result.web_search_main)
                
                # Key points in a card-like container
                st.markdown("## Key Points")
                with st.container():
                    for bullet in result.web_search_bullets.split('\n'):
                        if bullet.strip():
                            st.markdown(bullet)
                
                # Sources in an expander
                with st.expander("📚 Sources"):
                    for i, url in enumerate(result.visited_urls, 1):
                        st.markdown(f"{i}. [{url}]({url})")

# Footer
st.markdown("---")
