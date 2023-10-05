from sentence_transformers import SentenceTransformer
from torch import cosine_similarity

sentences = [
    "This is an example sentence",
    "Not a real sentence?!",
    "Each sentence is converted",
]

model = SentenceTransformer("IRI2070/dal-sbert-address-distilled-384-v2")
embeddings = model.encode(sentences)

# do something with the embeddings
# like calculating cosine similarities
for sent in sentences:
    print(sent)
    for sent2 in sentences:
        print(f"\t{sent2}")
        print(
            "\t",
            cosine_similarity(
                model.encode([sent], convert_to_numpy=False, convert_to_tensor=True),
                model.encode([sent2], convert_to_numpy=False, convert_to_tensor=True),
            ),
        )
