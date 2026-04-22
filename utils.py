from PIL import Image
from gtts import gTTS
import os
import tempfile

def load_and_preprocess_image(uploaded_file):
    """
    Loads an image from a file-like object and converts it to RGB.
    This ensures compatibility with the AI models.
    """
    try:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Could not process the image: {e}")

def text_to_speech(text):
    """
    Converts text to an audio file using Google Text-to-Speech.
    Returns the file path to the temporary audio file.
    """
    tts = gTTS(text=text, lang='en')
    # Create a temporary file to store the audio, saving disk space
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "caption_audio.mp3")
    tts.save(audio_path)
    return audio_path

def create_download_text(short_caption, detailed_caption, alterative_captions, keywords):
    """
    Formats the generated captions and keywords into a structured 
    string suitable for downloading as a .txt file.
    """
    content = f"--- AI Image Captions ---\n\n"
    content += f"[Short Caption]\n{short_caption.capitalize()}\n\n"
    
    content += f"[Detailed Caption]\n{detailed_caption.capitalize()}\n\n"
    
    content += "[Alternative Captions]\n"
    for i, cap in enumerate(alterative_captions, 1):
        content += f"{i}. {cap.capitalize()}\n"
    content += "\n"
    
    content += "[Detected Keywords / Objects]\n"
    for kw in keywords:
        # Get the primary label since ImageNet labels can be comma-separated
        label = kw['label'].split(',')[0].title() 
        score = kw['score'] * 100
        content += f"- {label} (Confidence: {score:.1f}%)\n"
        
    return content
