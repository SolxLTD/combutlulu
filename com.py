import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="👋🏻 Hi WelcomeTo Combot Your Offline Computer Chatbot",
    layout="centered"
)
@st.cache_data
def load_knowledge():
    topics, sentences = [], []
    with open("data.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if "|" in line:
                topic, sentence = line.split("|", 1)
                topics.append(topic)
                sentences.append(sentence)
    return topics, sentences

topics, knowledge_base = load_knowledge()


def get_best_match(user_question, sentences):
    corpus = sentences + [user_question]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    best_score = np.max(similarities)
    best_index = np.argmax(similarities)
    return best_score, best_index


def chatbot(user_input):
    user_input = user_input.lower().strip()

    if not user_input:
        return "Please type a computer-related question."

    for topic, sentence in zip(topics, knowledge_base):
        keywords = topic.split("_")
        if any(k in user_input for k in keywords):
            return sentence.capitalize()

    score, index = get_best_match(user_input, knowledge_base)

    if score >= 0.35:
        return knowledge_base[index].capitalize()

    if 0.2 <= score < 0.35:
        topic = topics[index].replace("_", " ")
        return (
            "🤔 I found something related, but your question is unclear.\n\n"
            f"Are you asking about **{topic}**?\n"
            "Try asking:\n"
            f"- What is {topic}?\n"
            f"- What does {topic} do?\n"
            f"- What are the types of {topic}?\n"
            f"- Define {topic}.\n"
            f"- What are the uses of {topic}?\n"
            f"- How does {topic} work?\n"
            f"- Give examples of {topic}."
        )

    return (
        "❌ I don't have this information in my local knowledge.\n\n"
        f"You asked: \"{user_input}\"\n"
        "Please:\n"
        "- Rephrase the question\n"
        "- Ask about one computer concept at a time\n"
        "- Use simple computer terms"
    )


if "history" not in st.session_state:
    st.session_state.history = []


user_question = st.text_input("Type your question:")

if st.button("Ask"):
    response = chatbot(user_question)
    st.session_state.history.append(("You", user_question))
    st.session_state.history.append(("Bot", response))


st.divider()

for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<span class='user'>🧑 You:</span> {text}", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='bot'>🤖 Bot:</span> {text}", unsafe_allow_html=True)

