import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model_path = '/Users/kunal/Desktop/model_t5'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Streamlit app
st.title("Text Summarizer App")

# User input
user_input = st.text_area("Enter the text you want to summarize:")

if st.button("Summarize"):
    if user_input:
        # Tokenize and summarize
        inputs = tokenizer("summarize: " + user_input, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")