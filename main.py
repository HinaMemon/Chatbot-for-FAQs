import tkinter as tk
import nltk
import string
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= SAFE STOPWORDS LOAD =================
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ================= LOAD OR CREATE FAQ FILE =================
FAQ_FILE = "faqs.json"

try:
    with open(FAQ_FILE, "r") as f:
        faqs = json.load(f)
except FileNotFoundError:
    faqs = {
        "What is your return policy?": "You can return products within 7 days.",
        "How can I contact support?": "You can contact support via email or phone.",
        "Do you offer home delivery?": "Yes, we provide home delivery all over Pakistan.",
        "What payment methods are accepted?": "We accept cash, debit card, and online transfer.",
        "Where are you located?": "We are located in Karachi, Pakistan."
    }

# ================= TEXT PREPROCESS (SAFE TOKENIZER) =================
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ================= TF-IDF VECTORIZER =================
def update_vectors():
    global questions, processed_questions, vectorizer, faq_vectors
    questions = list(faqs.keys())
    processed_questions = [preprocess(q) for q in questions]
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(processed_questions)

update_vectors()

# ================= CHATBOT LOGIC WITH LEARNING =================
SIMILARITY_THRESHOLD = 0.5

def chatbot(user_input):
    user_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_processed])

    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match = similarity.argmax()
    best_score = similarity[0][best_match]

    if best_score < SIMILARITY_THRESHOLD:
        return None  # Bot doesn't know
    else:
        return faqs[questions[best_match]]

# ================= TKINTER GUI =================
def send_message(event=None):
    user_text = entry.get().strip()
    if not user_text:
        return

    chat.config(state=tk.NORMAL)
    chat.insert(tk.END, f"You: {user_text}\n")
    entry.delete(0, tk.END)

    response = chatbot(user_text)
    if response:
        chat.insert(tk.END, f"Bot: {response}\n\n")
    else:
        chat.insert(tk.END, f"Bot: I don't know the answer. Can you teach me? (Type the answer below)\n\n")
        learn_button(user_text)  # Enable learning

    chat.config(state=tk.DISABLED)
    chat.see(tk.END)

def learn_button(question):
    # Temporarily override the Send button to save new answer
    def save_answer():
        answer = entry.get().strip()
        if not answer:
            return
        faqs[question] = answer
        with open(FAQ_FILE, "w") as f:
            json.dump(faqs, f, indent=4)
        update_vectors()
        chat.config(state=tk.NORMAL)
        chat.insert(tk.END, f"Bot: Thanks! I've learned something new.\n\n")
        chat.config(state=tk.DISABLED)
        chat.see(tk.END)
        entry.delete(0, tk.END)
        btn.config(command=send_message)  # Restore Send button
        root.bind("<Return>", send_message)

    btn.config(command=save_answer)
    root.bind("<Return>", lambda event: save_answer())

# ================= TKINTER GUI SETUP =================
root = tk.Tk()
root.title("FAQ Chatbot (NLP Based with Learning)")
root.geometry("520x450")
root.resizable(False, False)

chat = tk.Text(root, font=("Arial", 10), wrap=tk.WORD, state=tk.DISABLED)
chat.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat.config(state=tk.NORMAL)
chat.insert(tk.END, "Bot: Hello! Ask me any FAQ question 😊\n\n")
chat.config(state=tk.DISABLED)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(fill=tk.X, padx=10, pady=5)
entry.focus()

btn = tk.Button(root, text="Send", command=send_message)
btn.pack(pady=5)

root.bind("<Return>", send_message)

root.mainloop()
