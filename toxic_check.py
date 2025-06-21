from transformers import pipeline

# Load pre-trained toxic text classification model
classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

# Prompt user to input text to classify
text = input("Enter the sentence you want to classify: ")
result = classifier(text)

label = result[0]["label"]
score = result[0]["score"]

print(f"Label: {label}")
print(f"Score: {round(score * 100, 2)}%")

