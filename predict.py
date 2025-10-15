import torch
from PIL import Image
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from dataset import Vocabulary
import yaml
import os

# ------------------ CONFIG ------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------ LOAD VOCAB ------------------
# vocab was saved in checkpoint
checkpoint_path = os.path.join(config['training']['save_path'], "checkpoint_epoch8.pth")  # change epoch if needed
checkpoint = torch.load(checkpoint_path, map_location=device)
vocab_word2idx = checkpoint['vocab']
idx2word = {idx: word for word, idx in vocab_word2idx.items()}

# ------------------ MODEL ------------------
from model import EncoderCNN, DecoderRNN

vocab_size = len(vocab_word2idx)

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

# ------------------ CAPTION FUNCTION ------------------
def generate_caption(image_path, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        features = encoder(image)
        inputs = features.unsqueeze(1)  # start with image feature
        states = None
        caption_idx = []

        for _ in range(max_len):
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))  # [1, vocab_size]
            predicted = outputs.argmax(1)
            predicted_id = predicted.item()
            if idx2word[predicted_id] == "<EOS>":
                break
            caption_idx.append(predicted_id)
            inputs = decoder.embed(predicted).unsqueeze(1)  # [1,1,embed_size]

    caption_words = [idx2word[idx] for idx in caption_idx]
    return " ".join(caption_words)

# ------------------ TEST ------------------
test_image = "E:/Mini projects/img_cap/dataset/Images/3624327440_bef4f33f32.jpg"  # replace with any image
caption = generate_caption(test_image)
print("Generated Caption:", caption)

