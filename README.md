# AI Image Caption Generator

An interactive web application built with Python and Streamlit that uses state-of-the-art AI models to automatically generate meaningful captions for any uploaded image.

## Key Features

- **Upload Interface**: Clean layout for uploading JPG/PNG images and previewing them instantly.
- **AI Captions**: Uses Salesforce's BLIP (`blip-image-captioning-base`) to produce a concise 1-line caption AND a detailed, descriptive 2-3 line caption.
- **Alternative Captions**: Generates diverse alternative descriptions using sampling techniques.
- **Object Detection & Confidence Scores**: Utilizes Google's Vision Transformer (`vit-base-patch16-224`) to extract key objects, scenes, and keywords along with AI confidence scores (e.g., 90.5%).
- **Text-to-Speech (Bonus)**: Incorporates `gTTS` to read the generated short caption out loud automatically through the UI.
- **Regenerate (Bonus)**: Interactive button to generate fresh alternative texts for the given image.
- **Export Captions (Bonus)**: Option to download all insights cleanly formatted into a text file.
- **No Database Dependency**: The entire app runs strictly in-memory over inference modules keeping things fast and minimal.

## Project Structure
```text
/AI_Image_Caption_Generator
│── app.py             # Main Streamlit UI layout and routing
│── model.py           # Contains the CaptionGenerator class for AI models handling
│── utils.py           # Includes image parsing, Text-to-Speech, and export utilities
│── requirements.txt   # File declaring dependencies
```

## System Requirements
- OS: Windows, macOS, or Linux
- Environment: Python 3.8+ 

## Installation & Setup

1. **Navigate to the Project Directory**
Ensure you are in the directory containing `app.py`.
```bash
cd "path/to/AI_Image_Caption_Generator"
```

2. **Create a Virtual Environment (Recommended)**
Isolate your dependencies to avoid conflicts.
```bash
python -m venv venv
```

Activate the environment:
- On Windows: `venv\Scripts\activate`
- On macOS/Linux: `source venv/bin/activate`

3. **Install Dependencies**
Install all required libraries, including Streamlit and PyTorch:
```bash
pip install -r requirements.txt
```

4. **Launch the Application**
```bash
streamlit run app.py
```

## First Time Execution Notice
The application will download the pretrained Hugging Face AI models during its very first run (approximately 1.0 - 1.5 GB in total). Ensure you have a stable network connection. Successive runs will bypass the download step and execute much faster from the local cache.
