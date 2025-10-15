import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FlickrDataset
from collate_fn import collate_fn
import yaml
from model import EncoderCNN, DecoderRNN

# ------------------ CONFIG ------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ DATA ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = FlickrDataset(
    csv_file=config['data']['captions_file'],
    img_dir=config['data']['images_dir'],
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_idx=dataset.vocab.word2idx["<PAD>"])
)

vocab_size = len(dataset.vocab.word2idx)

# ------------------ MODEL ------------------
dropout = config['model'].get('dropout', 0)
num_layers = config['model'].get('num_layers', 1)
if num_layers == 1:
    dropout = 0.0  # avoid warning

encoder = EncoderCNN(embed_size=config['model']['embed_size']).to(device)
decoder = DecoderRNN(
    embed_size=config['model']['embed_size'],
    hidden_size=config['model']['hidden_size'],
    vocab_size=vocab_size,
    num_layers=num_layers,
    dropout=dropout
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word2idx["<PAD>"])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=config['training']['lr'])

# ------------------ CHECKPOINT SETUP ------------------
save_path = config['training']['save_path']
os.makedirs(save_path, exist_ok=True)

# Find latest checkpoint
def get_latest_checkpoint(folder):
    checkpoints = [f for f in os.listdir(folder) if f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
    return os.path.join(folder, checkpoints[-1])

start_epoch = 0
latest_ckpt = get_latest_checkpoint(save_path)

if latest_ckpt:
    print(f"Loading checkpoint from {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = int(latest_ckpt.split("epoch")[-1].split(".")[0])
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")

# ------------------ TRAIN LOOP ------------------
num_epochs = config['training']['epochs']

for epoch in range(start_epoch, num_epochs):
    for i, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)
        targets = captions
        outputs = outputs[:, :targets.size(1), :]

        # Compute loss
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(save_path, f"checkpoint_epoch{epoch+1}.pth")
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vocab': dataset.vocab.word2idx
    }, ckpt_path)
    print(f"✅ Saved checkpoint for epoch {epoch+1}")
