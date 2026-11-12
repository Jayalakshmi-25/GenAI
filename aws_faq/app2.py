
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import pygame
import os

# Step 1: Read Excel file
df = pd.read_excel("aws_faq.xlsx")

df = df[["Question", "Answer"]].dropna()

questions = df["Question"].astype(str).tolist()
answers = df["Answer"].astype(str).tolist()

# Step 2: Convert questions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
faq_vectors = vectorizer.fit_transform(questions)

# Step 3: Initialize audio player
pygame.mixer.init()

# Step 4: Save and play audio
def speak(answer):

    # Save response as MP3 file
    audio_file = "chatbot_response.mp3"

    tts = gTTS(text=answer, lang="en")
    tts.save(audio_file)

    print("Audio saved:", os.path.abspath(audio_file))

    # Play the saved MP3
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Wait until audio finishes
    clock = pygame.time.Clock()

    while pygame.mixer.music.get_busy():
        clock.tick(10)

    # Release audio file
    pygame.mixer.music.unload()


# Step 5: Chatbot function
def chatbot(user_question):

    user_vector = vectorizer.transform([user_question])

    scores = cosine_similarity(
        user_vector, faq_vectors
    )[0]

    best_index = np.argmax(scores)
    best_score = scores[best_index]

    print("Matched Question:", questions[best_index])
    print("Similarity Score:", round(best_score, 3))

    if best_score >= 0.2:
        answer = answers[best_index]
    else:
        answer = "Sorry, I could not find the answer."

    print("Bot:", answer)

    # Save MP3 and play it
    speak(answer)


# Step 6: Start chatbot
while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot stopped.")
        break

    if not user_input:
        continue

    chatbot(user_input)

pygame.mixer.quit()
