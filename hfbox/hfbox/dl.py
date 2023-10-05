from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from super_image import MsrnModel, ImageLoader

# from transformers import AutoModelForCausalLM,
# from sentence_transformers import SentenceTransformer
# from diffusers import StableDiffusionPipeline

#
# Oft-vital ML packages: safetensors, accelerate, transformers, xformers,
# datasets, sentence_transformers, diffusers
# last


model_dict = {
    "upscaling_models": [
        (MsrnModel, "eugenesiow/msrn-bam"),
        "stabilityai/stable-diffusion-x4-upscaler",
    ],
    "diffusion_models": [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0",  # base is text-to-image
        "stabilityai/stable-diffusion-xl-refiner-1.0",  # refiner is image-to-image
    ],
    "sentence_transformers": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/LaBSE",
        "IRI2070/dal-sbert-address-distilled-384-v2,",
    ],
    "language_models": [
        "distilgpt2",
        "xlnet-large-cased",
        # "xlnet-base-cased",
        "Malmuk1/cerebras_btlm-3b-8k-base_sharded",
        # "sgugger/sharded-gpt-j-6B",
        "vilsonrodrigues/falcon-7b-instruct-sharded",
        "vilsonrodrigues/falcon-7b-sharded",
        "codersasi/falcoder-7b-sharded-bf16",
        "TinyPixel/Llama-2-7B-bf16-sharded",
        # "guardrail/llama-2-7b-guanaco-instruct-sharded",
        "ethzanalytics/mpt-7b-storywriter-sharded",
        "Trelis/mpt-7b-8k-chat-sharded-bf16",
        # "hiepnh/vicuna-13B-1.1-HF-sharded",
        # "TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g",
        "ethzanalytics/blip2-flan-t5-xl-sharded",
    ],
    "bert_models": [
        "bert-base-uncased",
        "bert-base-cased",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "j-hartmann/emotion-english-distilroberta-base",
    ],
    "feature_extraction": [
        "facebook/bart-large",
        "eugenesiow/bart-paraphrase",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "microsoft/codebert-base",  # https://github.com/microsoft/CodeBERT
        "google/vit-base-patch16-224-in21k",
    ],
    "graph_ml": [
        # "Huhujingjing/custom-gcn",
    ],
}


all_models = [model for model_list in model_dict.values() for model in model_list]
# download with huggingface
for model in all_models:
    # hf_hub_download(model)
    print("Downloading model: ", model)
    download = True
    if isinstance(model, tuple):
        model_id = model[1]
        model_loader = model[0]
        try:
            model = model_loader.from_pretrained(model_id, trust_remote_code=True)
            download = False
        except:
            print("Failed to load model, attempting to download")
            model = model_id

    if download:
        try:
            pipeline(model=model, trust_remote_code=True)
        except:
            print("Not a pipeline, attempting to load model")
            try:
                AutoModel.from_pretrained(model, trust_remote_code=True)
            except:
                print("Complete failure to load model... skipping")
