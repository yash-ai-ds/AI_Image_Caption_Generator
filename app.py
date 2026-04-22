import streamlit as st
from PIL import Image
from model import CaptionGenerator
from utils import load_and_preprocess_image, text_to_speech, create_download_text

# Configure the Streamlit page layout and metadata
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling components
st.markdown("""
<style>
    .caption-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
        color: #1a1a1a;  /* Force dark text for visibility in dark mode */
    }
    .keyword-pill {
        display: inline-block;
        background-color: #e0e0e0;
        border-radius: 15px;
        padding: 5px 12px;
        margin: 5px;
        font-size: 14px;
        color: #333;
        font-weight: 500;
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the caption generator models and cache them in memory."""
    with st.spinner("Loading AI models (BLIP and ViT)... This may take a minute on first run."):
        return CaptionGenerator()

def main():
    st.title("✨ AI Image Caption Generator")
    st.markdown("Upload an image (JPG/PNG) to instantly generate meaningful descriptions, alternative options, and extract keywords!")

    # Initialize the models
    model = load_model()

    # Sidebar for uploading and controls
    with st.sidebar:
        st.header("Upload Center")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.markdown("### Generation Settings")
        num_alternatives = st.slider("Number of Alternative Captions", min_value=1, max_value=5, value=3)
        
        st.markdown("---")
        st.markdown("### System Architecture")
        st.info(
            "- **Captioning**: Salesforce/blip-image-captioning-base\n"
            "- **Object Detection**: google/vit-base-patch16-224\n"
            "- **Data Policy**: 100% In-Memory Processing, No Databases involved."
        )

    if uploaded_file is not None:
        try:
            # Process image to ensure correct format
            image = load_and_preprocess_image(uploaded_file)
            
            # Using Streamlit columns for a well-organised layout
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.image(image, caption="Uploaded Image", width='stretch')
                
                # Regenerate button to create new alternatives
                regenerate = st.button("🔄 Generate / Regenerate Captions", use_container_width=True)
                
            # If the user clicks generating or standard execution where it has not been generated for this image yet
            if regenerate or "captions" not in st.session_state or st.session_state.current_image != uploaded_file.name:
                with st.spinner("🧠 Analyzing image and drafting intelligent captions..."):
                    short_caption = model.generate_short_caption(image)
                    detailed_caption = model.generate_detailed_caption(image)
                    alt_captions = model.generate_alternative_captions(image, num_captions=num_alternatives)
                    keywords = model.extract_keywords(image)
                    
                    # Store generated data in session_state to avoid repeating predictions on UI interactions
                    st.session_state.captions = {
                        "short": short_caption,
                        "detailed": detailed_caption,
                        "alts": alt_captions,
                        "keywords": keywords
                    }
                    st.session_state.current_image = uploaded_file.name

            # Display generated data if it exists in session
            if "captions" in st.session_state and st.session_state.current_image == uploaded_file.name:
                caps = st.session_state.captions
                
                with col2:
                    st.subheader("📝 Generated Output")
                    
                    # Short Caption Section
                    st.markdown("**Short Caption**")
                    st.markdown(f'<div class="caption-box"><strong>{caps["short"].capitalize()}</strong></div>', unsafe_allow_html=True)
                    
                    # Text-to-Speech (Bonus Feature)
                    audio_path = text_to_speech(caps["short"])
                    st.audio(audio_path, format="audio/mp3")
                    
                    # Detailed Caption Section
                    st.markdown("**Detailed Caption**")
                    st.markdown(f'<div class="caption-box">{caps["detailed"].capitalize()}</div>', unsafe_allow_html=True)
                    
                    # Alternative Captions
                    st.markdown("**Alternative Options**")
                    for i, cap in enumerate(caps["alts"]):
                        st.markdown(f"- *{cap.capitalize()}*")
                    
                    st.markdown("---")
                    
                    # Keywords / Objects & Confidence Scores
                    st.subheader("🔍 Detected Objects & Keywords")
                    keyword_html = ""
                    for kw in caps["keywords"]:
                        # Prettify the label output
                        label = kw['label'].split(',')[0].title() 
                        score = kw['score'] * 100
                        keyword_html += f'<span class="keyword-pill">{label} ({score:.1f}%)</span>'
                    st.markdown(keyword_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Download functionality
                    download_text = create_download_text(caps["short"], caps["detailed"], caps["alts"], caps["keywords"])
                    st.download_button(
                        label="📥 Download Results as TXT",
                        data=download_text,
                        file_name="image_captions.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error processing the image! Issue: {str(e)}")

if __name__ == "__main__":
    main()
