from sentence_transformers import SentenceTransformer, util

#Create 10 sentences
sentences = [
    "The sun rises in the east.",
    "I love playing football.",
    "Artificial intelligence is transforming industries.",
    "Streamlit makes data apps easy to build.",
    "Python is a versatile programming language.",
    "The weather today is sunny and warm.",
    "Reading books expands knowledge.",
    "Music can improve your mood.",
    "Traveling helps you learn about cultures.",
    "Coffee keeps me awake during work."
]

#Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Generate embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute similarity between sentence pairs
# Ex:similarity between sentence 0 and 1
sim_0_1 = util.cos_sim(embeddings[0], embeddings[1])
print("Similarity between 0 and 1:", sim_0_1.item())

# Compare all pairs
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[j])
        print(f"Similarity({i}, {j}): {sim.item():.4f}")
