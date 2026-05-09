import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Page configuration
st.set_page_config(page_title="Yelp Review Classifier", page_icon="⭐")

@st.cache_resource
def load_model():
    """Load and cache the model/tokenizer to prevent reloading on every click"""
    model_name = "Bonnnz/CustomModel_yelp"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

# App UI
st.title("⭐ Yelp Review Sentiment Analyzer")
st.write("Enter a review below to predict the rating (1-5 stars).")

model, tokenizer = load_model()

# User Input
text = st.text_area(
    "Review Text:", 
    placeholder="Type your experience here...",
    height=200,
    value="dr. goldberg offers everything i look for in a general practitioner..." # Default from your script
)

if st.button("Predict Rating", type="primary"):
    if text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing sentiment..."):
            # Tokenization
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().detach().numpy()

            # Process result
            max_index = np.argmax(predictions)
            confidence = predictions[0][max_index]

            # Display Results
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Rating", f"{max_index + 1} Stars")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")

            # Visual star representation
            st.write("### Review Score")
            st.write("⭐" * (max_index + 1))
