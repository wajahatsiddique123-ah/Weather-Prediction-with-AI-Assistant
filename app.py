import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(page_title="Weather Forecast AI", page_icon="🌤️", layout="centered")
st.title("🌤️ AI Weather Forecast")
st.subheader("Powered by LangChain & Groq")

# Sidebar API Key Input
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)

# Function to get or construct the chain
def get_chain(api_key: str):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meteorologist and weather assistant. Provide helpful, structured, and insightful weather summaries and predictions based on the user's input."),
        ("human", "Provide a detailed weather forecast and recommendations for: {city}")
    ])

    return prompt | llm

# User Input
city = st.text_input("Enter City Name:", placeholder="e.g., Karachi, London, Tokyo")

if st.button("Get Forecast"):
    if not city.strip():
        st.warning("Please enter a city name.")
    elif not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        with st.spinner(f"Fetching weather insights for {city}..."):
            try:
                chain = get_chain(groq_api_key)
                response = chain.invoke({"city": city})
                st.success(f"Forecast for {city}")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Error invoking model: {str(e)}")
