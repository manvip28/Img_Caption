import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import yaml
import os
from model import EncoderCNN, DecoderRNN
import gdown

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Image Captioning", layout="wide", initial_sidebar_state="expanded")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
h1 {
    background: linear-gradient(120deg, #ffd89b 0%, #19547b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.sidebar .sidebar-content { background: linear-gradient(180deg, #4a5568 0%, #2d3748 100%); }
div[data-testid="stFileUploader"] { background-color: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; border: 2px dashed rgba(255,255,255,0.3); }
.stButton>button { width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; padding: 12px 24px; border-radius: 8px; border: none; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; }
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); }
div[data-testid="stImage"] { border-radius: 15px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3); overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------ CONFIG ------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------ DOWNLOAD CHECKPOINT FROM GOOGLE DRIVE ------------------
MODEL_PATH = "checkpoint_epoch8.pth"
FILE_ID = "1EtBH6Z3aedXYrcbR_iXAStMj7Bojvrj1"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model from Google Drive..."):
        try:
            # Use fuzzy=True for large files that might trigger virus scan
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            
            # Verify the file was downloaded and check size
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                st.success(f"✅ Model downloaded successfully! File size: {file_size / (1024**2):.2f} MB")
            else:
                st.error("❌ Download completed but file not found!")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Download failed: {type(e).__name__}")
            st.error(f"Error details: {str(e)}")
            st.info("💡 Please check: 1) Google Drive file permissions 2) File ID is correct 3) Internet connection")
            st.stop()
else:
    file_size = os.path.getsize(MODEL_PATH)
    st.info(f"ℹ️ Using cached model ({file_size / (1024**2):.2f} MB)")

# ------------------ LOAD CHECKPOINT ------------------
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load checkpoint: {type(e).__name__}")
    st.error(f"Error details: {str(e)}")
    st.info("""
    💡 Possible solutions:
    - Ensure you're using Python 3.11 (create `.python-version` file with `3.11`)
    - The checkpoint file might be corrupted, try deleting and re-downloading
    - Check if the file was saved with a compatible PyTorch version
    """)
    st.stop()

# Extract vocab and create mappings
vocab_word2idx = checkpoint['vocab']
idx2word = {idx: word for word, idx in vocab_word2idx.items()}
vocab_size = len(vocab_word2idx)

# ------------------ MODEL ------------------
try:
    encoder = EncoderCNN(embed_size=config['model']['embed_size']).to(device)
    decoder = DecoderRNN(
        embed_size=config['model']['embed_size'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=vocab_size,
        num_layers=config['model'].get('num_layers', 1),
        dropout=config['model'].get('dropout', 0)
    ).to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()
    
    st.success("✅ Model architecture initialized!")
    
except Exception as e:
    st.error(f"❌ Failed to initialize model: {type(e).__name__}")
    st.error(f"Error details: {str(e)}")
    st.stop()

# ------------------ CAPTION FUNCTION ------------------
def generate_caption(image: Image.Image, max_len=20):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image)
        inputs = features.unsqueeze(1)
        states = None
        caption_idx = []

        for _ in range(max_len):
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            predicted_id = predicted.item()
            if idx2word[predicted_id] == "<EOS>":
                break
            caption_idx.append(predicted_id)
            inputs = decoder.embed(predicted).unsqueeze(1)

    caption_words = [idx2word[idx] for idx in caption_idx if idx2word[idx] not in ["<SOS>", "<PAD>"]]
    return " ".join(caption_words)

# ------------------ STREAMLIT UI ------------------
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0; color:white'>🎨 AI Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 10px;'>Transform your images into words with deep learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("### 📤 Upload & Generate")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
#### 📋 How to use:
1. **Upload** a JPG, JPEG, or PNG image
2. Click **Generate Caption** button
3. View your AI-generated caption!

#### ℹ️ About:
This app uses a CNN-LSTM architecture to automatically generate descriptive captions for images.
""")

# Main content
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h3 style='color: white; text-align: center;'>📷 Your Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("<h3 style='color: white; text-align: center;'>✨ Generated Caption</h3>", unsafe_allow_html=True)
        
        if st.sidebar.button("🚀 Generate Caption"):
            with st.spinner("🤖 AI is analyzing your image..."):
                try:
                    caption = generate_caption(image)
                    
                    st.markdown(
                        f"""
                        <div style='
                            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,240,255,0.95) 100%);
                            border: none; padding: 30px; border-radius: 15px;
                            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3); margin-top: 20px;
                        '>
                            <h4 style='color: #667eea; margin-bottom: 15px; font-weight: 600;'>📝 Caption Result:</h4>
                            <p style=' font-size: 20px; color: #2d3748; line-height: 1.6; font-weight: 500; font-style: italic;'>" {caption} "</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"❌ Failed to generate caption: {type(e).__name__}")
                    st.error(f"Error details: {str(e)}")
        else:
            st.markdown(
                """
                <div style='background: rgba(255,255,255,0.1); border: 2px dashed rgba(255,255,255,0.3); padding: 40px; border-radius: 15px; text-align: center; margin-top: 20px;'>
                    <p style='color: rgba(255,255,255,0.8); font-size: 18px; margin: 0;'>👈 Click the button in the sidebar to generate a caption</p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); border: 2px dashed rgba(255,255,255,0.3); padding: 60px; border-radius: 20px; text-align: center; margin: 50px auto; max-width: 600px;'>
            <h2 style='color: white; margin-bottom: 20px;'>🖼️ No Image Uploaded</h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 18px;'>Please upload an image from the sidebar to get started</p>
        </div>
    """, unsafe_allow_html=True)