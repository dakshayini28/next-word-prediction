import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final.h5')

@st.cache_data
def load_tokenizer():
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
        text_data = file.read()
    tokenizer.fit_on_texts([text_data])
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

max_sequence_len = 18 

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

def display_predictions():
    st.title("Next Word Prediction")

    seed_text = st.text_input("Enter your text:", value="")
    
    next_words = st.number_input("Enter the number of words to predict:", min_value=1, max_value=500, value=1)
    
    if st.button("Predict"):
        if not seed_text:
            st.error("Please enter text.")
        else:
            result = generate_text(seed_text, next_words)
            st.markdown(f"<h3 style='font-size:20px; color:grey;'>Predicted Text: {result}</h3>", unsafe_allow_html=True)

def display_instructions():
    st.title("Instructions")
    st.write("""
    This app generates predicted text based on the seed input you provide. 
    - Enter some starting text in the "Enter your text" box.
    - Specify how many words you want to predict.
    - Click "Predict" to see the generated text.
    """)

def display_about_us():
    st.title("About Us")
    st.write("""
    This application was developed by a group of enthusiasts working on AI-powered text generation.
    Our goal is to showcase how machine learning models can be used for creative writing and text generation.
    """)

def main():
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select a page", ["Predictions", "Instructions", "About Us"])
    
    if page == "Predictions":
        display_predictions()
    elif page == "Instructions":
        display_instructions()
    elif page == "About Us":
        display_about_us()

if __name__ == "__main__":
    main()
