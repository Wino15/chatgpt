import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQChatbot:
    def __init__(self, faq_file):
        self.faqs = self.load_faqs(faq_file)
        self.questions = list(self.faqs.keys())
        self.answers = list(self.faqs.values())
        self.vectorizer = TfidfVectorizer().fit_transform(self.questions)

    def load_faqs(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            faqs = {item['question']: item['answer'] for item in data['faqs']}
            return faqs

    def generate_response(self, user_input):
        user_vector = TfidfVectorizer().fit(self.questions).transform([user_input])
        similarity_scores = cosine_similarity(user_vector, self.vectorizer).flatten()
        best_match_index = similarity_scores.argmax()
        if similarity_scores[best_match_index] > 0.3:  # Set a threshold for similarity
            return self.answers[best_match_index]
        else:
            return "I'm sorry, I don't have information on that topic. Please ask another question."

# Initialize the chatbot with the FAQs file
chatbot = FAQChatbot('faqs.json')

# Create the Streamlit app
st.title("Labour Law Chatbot")
st.write("Enter your question below:")

user_input = st.text_input("User Input")
if st.button("Submit"):
    response = chatbot.generate_response(user_input)
    st.write("Chatbot Response:", response)




