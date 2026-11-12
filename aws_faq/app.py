import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_excel("aws_faq.xlsx")
df = df.dropna(subset=["Question", "Answer"])
print(df.shape)
print(df.isna().sum())
questions=df['Question'].astype(str).tolist()
answers=df['Answer'].astype(str).tolist()
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)
def chatbot(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, question_vectors)
    most_similar_index = np.argmax(similarity_scores)
    return answers[most_similar_index]

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot(user_input)
    print("Chatbot:", response)