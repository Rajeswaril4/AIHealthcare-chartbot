import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# Load Question-Answering Model
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# Define context (you can expand this with more medical info)
context = """
Healthcare chatbots provide guidance on general health issues but do not replace professional medical advice. 
If you have severe symptoms, consult a doctor immediately. 
To book an appointment, provide your name, preferred date, and contact information.
"""

# Function to preprocess user input
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    filtered_words = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    return " ".join(filtered_words)

# Intent-based response handling
def healthcare_chatbot(user_input):
    processed_input = preprocess_text(user_input)

    if "symptom" in processed_input or "pain" in processed_input:
        return "I'm not a doctor, but you should consider consulting a healthcare professional for accurate diagnosis."
    elif "appointment" in processed_input or "schedule" in processed_input:
        return "Would you like to book an appointment? Please provide your name and preferred date."
    else:
        try:
            response = qa_pipeline(question=user_input, context=context)
            return response["answer"]
        except Exception:
            return "I'm sorry, I couldn't process your request. Can you please rephrase?"

# Streamlit UI
def main():
    st.title("🩺 Healthcare Assistant Chatbot")
    st.write("Ask me about symptoms, appointments, or general healthcare queries.")

    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("👤 **User:**", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
            st.write("🤖 **Healthcare Assistant:**", response)
        else:
            st.warning("Please enter a message to get a response.")

if __name__ == "__main__":
    main()
