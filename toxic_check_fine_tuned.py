from transformers import pipeline

# Load the fine-tuned model and tokenizer from local folder
classifier = pipeline("text-classification", model="./finetuned-toxic-roberta", tokenizer="./finetuned-toxic-roberta")

# Prompt user for input
text = input("Enter the sentence you want to classify: ")
result = classifier(text)

label = result[0]["label"]  # 'LABEL_0' or 'LABEL_1'
score = result[0]["score"]

if label == "LABEL_1":
    print(f"Label: Toxic")
else:
    print(f"Label: Not Toxic")
print(f"Score: {round(score * 100, 2)}%")

