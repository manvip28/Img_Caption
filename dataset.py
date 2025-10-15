import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import spacy
from collections import Counter
import nltk

# download NLTK punkt if not already
nltk.download('punkt')
spacy_en = spacy.load("en_core_web_sm")

# ------------------ Vocabulary ------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx["<PAD>"] = 0
        self.word2idx["<SOS>"] = 1
        self.word2idx["<EOS>"] = 2
        self.word2idx["<UNK>"] = 3
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4  # starting index for real words

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in tokenized_text
        ]

# ------------------ Dataset ------------------
class FlickrDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, freq_threshold=5):
        """
        csv_file: path to captions CSV, with columns ['image','caption']
        img_dir: path to images folder
        """
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.df['caption'].dropna().tolist())

        # store list of dicts for fast access, only include valid images
        self.data = []
        for idx, row in self.df.iterrows():
            img_name = str(row['image']).strip()
            caption = str(row['caption']).strip()
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: Image not found, skipping: {img_path}")
                continue
            if len(caption) == 0:
                continue  # skip empty captions
            self.data.append({
                'image': img_name,
                'caption': caption
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])

        # Open image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Warning: Failed to open image, skipping: {img_path}")
            return None

        if self.transform is not None:
            image = self.transform(image)

        caption = item['caption']
        numericalized_caption = [self.vocab.word2idx["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx["<EOS>"])

        return image, torch.tensor(numericalized_caption)
