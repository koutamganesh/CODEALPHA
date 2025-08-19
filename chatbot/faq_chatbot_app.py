import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Collect FAQs (questions + answers)
faqs = {
    "How can I reset my password?": "You can reset your password by clicking 'Forgot Password' on the login page.",
    "How do I track my order?": "You can track your order using the tracking link sent to your email after shipping.",
    "What is the return policy?": "You can return products within 30 days of delivery.",
    "hi?": "Hi who can i help you.",
    "How do I contact customer support?": "You can reach us at support@example.com.",
    "Do you offer international shipping?": "Yes, we offer international shipping to selected countries."
    
}

# 2. Preprocess + Vectorize questions
questions = list(faqs.keys())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# 3. Match user query with closest FAQ
def chatbot_response(user_query):
    user_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return faqs[questions[index]]

# 4. Run chatbot loop
def faq_chatbot():
    print("Hello! I am your FAQ Chatbot . Ask me a question!")
    while True:
        user = input("You: ")
        if user.lower() in ["bye", "exit", "quit"]:
            print("Chatbot: Goodbye! ")
            break
        response = chatbot_response(user)
        print("Chatbot:", response)

# 5. Start
faq_chatbot()
