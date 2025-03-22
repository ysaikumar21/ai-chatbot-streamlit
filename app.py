import streamlit as st
import nltk
import numpy as np
import pandas as pd
import random
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from nltk.corpus import stopwords

# ✅ Step 3: Download NLTK Data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')  # ADD THIS LINE temporarily


# ✅ Step 4: Initialize Lemmatizer & Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ✅ Step 5: Define Updated Intents (More Examples)
intents = [
    {
        "intent": "greetings",
        "patterns": ["Hello", "Hi", "Good day", "Hey", "How are you?", "Good morning", "Hi there", "Greetings", "Hey bot"],
        "responses": [
            "Hello! Welcome to Social Prachar Institute. How can I assist you today?",
            "Hi there! How can I help you with your queries about our courses?",
            "Good day! I’m Vajra.AI, your guide to the Social Prachar Institute. What would you like to know?"
        ]
    },
    {
        "intent": "farewells",
        "patterns": ["Goodbye", "Bye", "Thank you", "See you later", "Take care", "Farewell", "Thanks", "See you soon"],
        "responses": [
            "Thank you for visiting! Have a great day!",
            "It was my pleasure to assist you. Take care and good luck with your learning journey!",
            "Goodbye, and feel free to come back if you have more questions!"
        ]
    },
    {
        "intent": "fee_structure",
        "patterns": ["How much does the course cost?", "What are the fees?", "Tell me the course fees.", "Price of Data Science?", "Tuition fees?"],
        "responses": [
            "The fee for Data Science is 50k, and other courses are 30k.",
            "Our courses are priced as follows: Data Science - 50k, Data Analytics - 30k, Python Full Stack - 30k, Java Full Stack - 30k, AWS Developer - 30k."
        ]
    },
    {
        "intent": "course_info",
        "patterns": ["What courses do you offer?", "Tell me about courses", "List all courses", "Available programs?", "Which subjects?", "Training programs?", "Social Prachar courses?"],
        "responses": [
            "We offer Data Science, Data Analytics, Python Full Stack, Java Full Stack, and AWS Developer courses.",
            "Our courses include Data Science & AI, Data Analytics, Python Full Stack, Java Full Stack, and AWS Developer. You can learn more on our website."
        ]
    }
]

# ✅ Step 6: Preprocess Data
def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

training_sentences = []
training_labels = []
class_names = []

for intent in intents:
    for pattern in intent["patterns"]:
        processed_sentence = preprocess_sentence(pattern)
        training_sentences.append(processed_sentence)
        training_labels.append(intent["intent"])

    if intent["intent"] not in class_names:
        class_names.append(intent["intent"])

# ✅ Step 7: Feature Extraction (Optimized TF-IDF)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=7000, min_df=2)
X = vectorizer.fit_transform(training_sentences).toarray()

# ✅ Step 8: Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(training_labels)

# ✅ Step 9: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) # Increased test_size to 0.2

# ✅ Step 10: Train **XGBoost Classifier** (Higher Accuracy)
xgb_model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# ✅ Step 11: Evaluate the Model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'✅ Model Training Complete! Accuracy: {accuracy:.3f}')

# ✅ Step 12: Save Model & Vectorizer
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("✅ Model & vectorizer saved! Ready for chatbot deployment.")

# ✅ Step 13: Load Model for Deployment
@st.cache_resource
def load_model():
    with open("chatbot_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return model, vectorizer, label_encoder

xgb_model, vectorizer, label_encoder = load_model()

# ✅ Step 14: Prediction Function (Higher Confidence Threshold)
def predict_intent(user_input):
    user_input_tokens = preprocess_sentence(user_input)
    input_vector = vectorizer.transform([user_input_tokens]).toarray()
    prediction_prob = xgb_model.predict_proba(input_vector)

    predicted_class_index = np.argmax(prediction_prob)
    confidence = np.max(prediction_prob)

    if confidence < 0.2:
        return "I'm not certain. Could you clarify your question?"

    prediction = label_encoder.classes_[predicted_class_index]

    for intent in intents:
        if intent['intent'] == prediction:
            return random.choice(intent['responses'])

    return "Sorry, I didn’t understand that. Could you please rephrase?"

# ✅ Step 15: Chatbot Interaction
if __name__ == "__main__":
    st.title("Vajra.AI Chatbot")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Type your message..."):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the chatbot's response
        response = predict_intent(prompt)

        # Display the chatbot's response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)