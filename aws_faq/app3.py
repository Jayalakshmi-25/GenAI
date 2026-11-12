#api="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"
import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Step 2: AWS FAQ dataset
faq = {
    "What is Amazon EC2?":
        "Amazon EC2 provides virtual computing servers in AWS.",

    "What is Amazon S3?":
        "Amazon S3 is an object storage service.",

    "What is AWS Lambda?":
        "AWS Lambda runs code without managing servers.",

    "What is Amazon RDS?":
        "Amazon RDS is a managed relational database service.",

    "What is AWS IAM?":
        "AWS IAM manages users, roles and permissions.",

    "What is Amazon VPC?":
        "Amazon VPC provides an isolated virtual network.",

    "What is Amazon CloudWatch?":
        "CloudWatch monitors AWS applications and resources."
}

questions = list(faq.keys())
answers = list(faq.values())

# Step 3: Generate embeddings for all FAQ questions
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=questions
)

faq_embeddings = np.array([
    item.embedding for item in response.data
])

print("FAQ Embeddings Shape:", faq_embeddings.shape)

# Step 4: Chatbot function
def chatbot(sentence):

    # Convert user question into embedding
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=sentence
    )

    user_embedding = np.array(
        response.data[0].embedding
    ).reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        user_embedding,
        faq_embeddings
    )[0]

    # Find highest similarity
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    print("\nUser Question:", sentence)
    print("Matched Question:", questions[best_index])
    print("Cosine Similarity:", round(best_score, 4))

    # Return matching answer
    if best_score >= 0.5:
        print("Bot:", answers[best_index])
    else:
        print("Bot: Sorry, I don't know the answer.")

# Step 5: Test chatbot
chatbot("amazon ec2 provides computing services")

chatbot("Where can I store files in AWS?")

chatbot("How do I run code without servers?")
