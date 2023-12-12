import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import requests

# Load model and tokenizer
model_path = '/Users/kunal/Desktop/model_t5'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Streamlit app
st.title("Text Summarizer App")

# User input
user_input_url = st.text_input("Enter the URL of the text you want to summarize:")

if st.button("Summarize"):
    if user_input_url:
        try:
            # Fetch content from the provided URL
            response = requests.get(user_input_url)
            response.raise_for_status()
            content = response.text
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching content from the provided URL: {e}")
            st.stop()

        # Tokenize and summarize
        inputs = tokenizer("summarize: " + content, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter the URL of the text you want to summarize.")
