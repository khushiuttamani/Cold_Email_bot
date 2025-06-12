import os
import uuid
import pandas as pd
import chromadb
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"  # New recommended model
)

# Prompt templates
prompt_extract = PromptTemplate.from_template("""
### SCRAPED TEXT FROM WEBSITE:
{page_data}
### INSTRUCTION:
The scraped text is from the career's page of a website.
Your job is to extract the job postings and return them in JSON format containing the 
following keys: `role`, `experience`, `skills`, and `description`.
Only return the valid JSON.
### VALID JSON (NO PREAMBLE):    
""")

prompt_email = PromptTemplate.from_template("""
### JOB DESCRIPTION:
{job_description}

### INSTRUCTION:
You are a Computer Science Engineering student working on an AI-powered cold email generator project.
The goal of this project is to demonstrate how artificial intelligence can automate and personalize outreach based on job descriptions.

Your task is to generate a professional and relevant cold email, showcasing how your technical knowledge, experience with AI tools, and familiarity with projects (referenced in the following links: {link_list}) can add value to the client’s needs.

Write a short, clear, and compelling cold email expressing genuine interest and aligning your academic and project experience with the job role.

Do not provide a preamble.

### EMAIL (NO PREAMBLE):
""")


# Cache portfolio loading using st.cache_resource
@st.cache_resource
def load_portfolio():
    df = pd.read_csv("my_portfolio.csv")
    client = chromadb.PersistentClient("vectorstore")
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(
                documents=[row["Techstack"]],
                metadatas=[{"links": row["Links"]}],
                ids=[str(uuid.uuid4())]
            )
    return collection

# Streamlit UI
st.title("Cold Email Generator")
st.markdown("Enter a job page URL and get a custom cold email generated using AI.")

job_url = st.text_input("Enter the job URL:")

if st.button("Generate Email") and job_url:
    with st.spinner("🔍 Scraping and generating cold email..."):
        try:
            # Step 1: Load job content
            loader = WebBaseLoader(job_url)
            page_data = loader.load().pop().page_content

            # Display scraped content preview
            st.text_area("🔎 Scraped Page Preview", value=page_data[:1500], height=300)

            if not page_data or len(page_data.strip()) < 100:
                st.error("❌ Scraped content is empty or too short. The website might block scraping or use JavaScript.")
                st.stop()

            # Step 2: Extract job info
            chain_extract = prompt_extract | llm
            res = chain_extract.invoke({"page_data": page_data})
            json_parser = JsonOutputParser()

            try:
                job = json_parser.parse(res.content)
                if isinstance(job, list):
                    job = job[0]
            except Exception:
                st.error("❌ Invalid JSON from model:\n\n" + res.content)
                st.stop()

            # Step 3: Query ChromaDB with skills
            collection = load_portfolio()
            query_text = job.get("skills", "")
            links = collection.query(query_texts=query_text, n_results=2).get("metadatas", [])
            formatted_links = [meta["links"] for meta in links if "links" in meta]

            # Step 4: Generate cold email
            chain_email = prompt_email | llm
            email_res = chain_email.invoke({
                "job_description": str(job),
                "link_list": formatted_links
            })

            # Output final email
            st.subheader("📧 Cold Email")
            st.code(email_res.content, language="markdown")

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
