# 🚀 Cold Email Generator using LangChain, Groq LLM, and ChromaDB

This project automates the generation of customized cold emails based on job descriptions scraped from company career pages. It extracts key job details using a Groq-hosted LLM, matches them with relevant case studies from a portfolio using ChromaDB, and generates a personalized outreach email—all through a clean Streamlit interface.

---

## 🔧 Features

- 🔗 Accepts a job URL as input  
- 🧠 Extracts job role, experience, skills, and description using LLM  
- 🔍 Matches required skills with company portfolio using ChromaDB vector search  
- ✉️ Generates professional cold emails based on AtliQ’s profile  
- 🖥️ Streamlit web interface for easy interaction  

---

## 📁 Project Structure

Cold_Email_Generator/
├── app.py # Main Streamlit app
├── my_portfolio.csv # Portfolio with techstack and links
├── requirements.txt # Python dependencies
├── .env # API key for Groq
├── vectorstore/ # ChromaDB persistent storage

Required Libraries:
- streamlit
- langchain
- langchain-groq
- chromadb
- pandas
- python-dotenv
- beautifulsoup4
- unstructured

## Environment Setup
Create a .env file and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

## Run the App
streamlit run app.py

