import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

class CaptionGenerator:
    def __init__(self):
        """
        Initializes the AI models for captioning and object detection.
        Keeps everything contained in memory.
        """
        # Determine whether to use GPU or CPU. 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load BLIP for generating text captions from images
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        # Load ViT (Vision Transformer) for object detection / keyword extraction
        # This will output ImageNet classes which serve as excellent keywords.
        self.classifier = pipeline(
            "image-classification", 
            model="google/vit-base-patch16-224", 
            device=0 if self.device == "cuda" else -1
        )
        
    def generate_short_caption(self, image):
        """Generates a concise, 1-line caption."""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
        return self.processor.decode(out[0], skip_special_tokens=True)
        
    def generate_detailed_caption(self, image):
        """Generates a more detailed caption."""
        # Let the model generate naturally but enforce a longer minimum length
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            min_new_tokens=20,
            max_new_tokens=80,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
        return self.processor.decode(out[0], skip_special_tokens=True).strip()
        
    def generate_alternative_captions(self, image, num_captions=3):
        """Generates multiple alternative captions using sampling."""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # Using sampling to get diverse alternative descriptions
        out = self.model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=num_captions
        )
        return [self.processor.decode(o, skip_special_tokens=True) for o in out]
        
    def extract_keywords(self, image, top_k=5):
        """Extracts objects/keywords and confidence scores from the image."""
        predictions = self.classifier(image)
        # Predictions is a list of dicts: [{'score': 0.9, 'label': 'cat'}, ...]
        return predictions[:top_k]
