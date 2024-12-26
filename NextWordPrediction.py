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

def main():
    st.title("Next Word Prediction")

    seed_text = st.text_input("Enter your text:", value="")
    
    next_words = st.number_input("Enter the number of words to predict:", min_value=1, max_value=100, value=1)
    
    if st.button("Predict"):
        if not seed_text:
            st.error("Please enter text.")
        else:
            result = generate_text(seed_text, next_words)
            st.markdown(f"<h3 style='font-size:24px; color:black;'>Predicted Text: {result}</h3>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
