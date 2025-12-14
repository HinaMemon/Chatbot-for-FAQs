import tkinter as tk
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= SAFE STOPWORDS LOAD =================
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ================= FAQs =================
faqs = {
    "What is your return policy?": "You can return products within 7 days.",
    "How can I contact support?": "You can contact support via email or phone.",
    "Do you offer home delivery?": "Yes, we provide home delivery all over Pakistan.",
    "What payment methods are accepted?": "We accept cash, debit card, and online transfer.",
    "Where are you located?": "We are located in Karachi, Pakistan."
}

# ================= TEXT PREPROCESS (NO NLTK TOKENIZER) =================
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()   # SAFE tokenizer
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ================= PREPARE DATA =================
questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# ================= CHATBOT LOGIC =================
def chatbot(user_input):
    user_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_processed])

    similarity = cosine_similarity(user_vector, faq_vectors)
    best_match = similarity.argmax()

    return faqs[questions[best_match]]

# ================= TKINTER GUI =================
def send_message():
    user_text = entry.get().strip()
    if not user_text:
        return

    chat.insert(tk.END, f"You: {user_text}\n")
    response = chatbot(user_text)
    chat.insert(tk.END, f"Bot: {response}\n\n")

    entry.delete(0, tk.END)
    chat.see(tk.END)

root = tk.Tk()
root.title("FAQ Chatbot (NLP Based)")
root.geometry("520x420")
root.resizable(False, False)

chat = tk.Text(root, font=("Arial", 10), wrap=tk.WORD)
chat.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat.insert(tk.END, "Bot: Hello! Ask me any FAQ question 😊\n\n")

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(fill=tk.X, padx=10, pady=5)

btn = tk.Button(root, text="Send", command=send_message)
btn.pack(pady=5)

root.bind("<Return>", lambda event: send_message())

root.mainloop()
