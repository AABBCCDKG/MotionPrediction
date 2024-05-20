import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import Generator, Discriminator
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_data_in_batches, extract_tarfile
from sklearn.preprocessing import LabelEncoder

def visualize_data(frames, num_images=5):
    plt.figure(figsize=(10, 2 * num_images))
    for i in range(num_images):
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(frames[i].astype(np.uint8))
        plt.axis('off')
    plt.show()

def train_cgan(generator, discriminator, data_loader, epochs=100, batch_size=32, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (real_images, *labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            optimizer_d.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1)
            fake_images = generator(noise)
            print(f'Fake images shape: {fake_images.shape}')  # 打印生成器生成的假图像形状
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()

            d_loss = d_loss_real + d_loss_fake

            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    torch.save(generator.state_dict(), '/content/drive/My Drive/generator.pth')
    torch.save(discriminator.state_dict(), '/content/drive/My Drive/discriminator.pth')
    print("Models saved as 'generator.pth' and 'discriminator.pth' in Google Drive")

def encode_non_numeric_data(labels):
    label_encoders = {}
    encoded_labels = {}
    
    for key in labels:
        if isinstance(labels[key][0], (np.str_, str)):
            le = LabelEncoder()
            encoded_labels[key] = le.fit_transform(labels[key].astype(str))
            label_encoders[key] = le
        else:
            encoded_labels[key] = labels[key]
    
    return encoded_labels, label_encoders

def ensure_size_match(frames, labels):
    min_length = min(frames.shape[0], min(len(v) for v in labels.values()))
    
    frames = frames[:min_length]
    for key in labels:
        labels[key] = labels[key][:min_length]
    
    return frames, labels

if __name__ == "__main__":
    tar_path = '/content/drive/My Drive/Penn_Action.tar.gz'
    extract_path = '/content/'
    frames_folder = os.path.join(extract_path, 'Penn_Action/frames')
    labels_folder = os.path.join(extract_path, 'Penn_Action/labels')

    extract_tarfile(tar_path, extract_path)

    all_frames = []
    all_labels = []
    batch_size = 50
    max_sequences = 100
    sequence_count = 0

    for batch_data in preprocess_data_in_batches(frames_folder, labels_folder, batch_size, target_size=(128, 128)):
        for frames, labels in batch_data:
            if sequence_count >= max_sequences:
                break
            all_frames.append(frames)
            if labels and len(labels) > 0:
                all_labels.append(labels)
            sequence_count += 1
        if sequence_count >= max_sequences:
            break

    all_frames = np.concatenate(all_frames, axis=0)
    all_labels = [label for label in all_labels if len(label) > 0]
    concatenated_labels = {key: np.concatenate([d[key] for d in all_labels], axis=0) for key in all_labels[0]}

    encoded_labels, label_encoders = encode_non_numeric_data(concatenated_labels)

    all_frames, encoded_labels = ensure_size_match(all_frames, encoded_labels)

    visualize_data(all_frames)

    preprocessed_frames = torch.tensor(all_frames.transpose((0, 3, 1, 2)), dtype=torch.float32)
    
    preprocessed_labels_dict = {}
    for key in encoded_labels:
        if isinstance(encoded_labels[key][0], (np.float32, np.float64)):
            preprocessed_labels_dict[key] = torch.tensor(encoded_labels[key], dtype=torch.float32)
        else:
            preprocessed_labels_dict[key] = torch.tensor(encoded_labels[key].astype(np.int64), dtype=torch.long)

    dataset = TensorDataset(preprocessed_frames, *preprocessed_labels_dict.values())

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()

    train_cgan(generator, discriminator, data_loader, epochs=100, batch_size=32, lr=0.0002)
