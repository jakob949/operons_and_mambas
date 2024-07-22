# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mambapy.mamba import Mamba, MambaConfig
# import sys
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"> Using {device} device")

# binary_classification = True

# def tokenizer(inputs):
#     token = {"A": 0, "T": 1, "C": 2, "G": 3, "P": 4}
#     if isinstance(inputs[0], str):

#         return [[token[j] for j in i.upper()] for i in inputs]
#     return inputs  # For binary labels, return as is

# def detokenizer(token):
#     detoken = {0: "A", 1: "T", 2: "C", 3: "G", 4: "P"}
#     if isinstance(token, torch.Tensor):
#         return detoken[token.item()]
#     return detoken[token]

# def get_batch(data_inputs, data_labels, batch_size):
#     indices = torch.randperm(len(data_inputs))[:batch_size]
#     input_batch = [data_inputs[i] for i in indices]
#     label_batch = [data_labels[i] for i in indices]
#     return torch.tensor(tokenizer(input_batch)), torch.tensor(label_batch), indices

# class Dataset:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.inputs = []
#         self.labels = []
    
#     def read_data(self):
#         with open(self.file_path, 'r') as file:
#             lines = file.readlines()
        
#         for line in lines:
#             input_seq, label = line.strip().split('\t')
#             if len(input_seq) < 2500:
#                 self.inputs.append(input_seq)
#                 self.labels.append(label)

#     def pad_sequences(self):
#         max_len_in = max(len(i) for i in self.inputs)
#         self.inputs = [i + "P" * (max_len_in - len(i)) for i in self.inputs]


# test = Dataset(file_path="test.txt")
# test.read_data()
# test.pad_sequences()
# data_inputs = tokenizer(test.inputs)[:50]

# train = Dataset(file_path="train.txt")
# train.read_data()
# train.pad_sequences()
# data_labels = tokenizer(train.inputs)[:50]

# max_len_in = len(data_inputs[0])



# # # Expand the dataset
# # data_inputs = ["ATTCAA", "ATTT", "CTTC", "CTTA", "GTTACGCGCGCGCG", "GTTG", "AATC", "CCTA", "GGAT", "ATAC"]
# # if binary_classification:
# #     data_labels = [1, 1, 0, 0, 0, 0, 1, 0, 0, 1]
# # else:
# #     data_labels = ["TAAGTT", "TAAA", "GAAG", "GAAT", "CAATGCGCGCGCGC", "CAAC", "TTAG", "GGAT", "CCTA", "TATG"]
# #
# # max_len_in = max(len(i) for i in data_inputs)
# # # apply padding
# # data_inputs = [i + "P" * (max_len_in - len(i)) for i in data_inputs]
# #
# # if not binary_classification:
# #     max_len_out = max(len(i) for i in data_labels)
# #     data_labels = [i + "P" * (max_len_out - len(i)) for i in data_labels]

# # print(f"> Inputs: {data_inputs}")



# class MambaWithProj(nn.Module):
#     def __init__(self, config, binary_classification):
#         super().__init__()
#         self.mamba = Mamba(config)
#         self.binary_classification = binary_classification
#         if binary_classification:
#             self.proj = nn.Linear(config.d_model, 1)
#         else:
#             self.proj = nn.Linear(config.d_model, 5)  # 5 classes: A, T, C, G, P
    
#     def forward(self, x):
#         x = self.mamba(x)
#         if self.binary_classification:
#             # Global average pooling
#             x = x.mean(dim=1)
#         return self.proj(x).squeeze(-1) if self.binary_classification else self.proj(x)

# def train():
#     epochs = 201
#     batch_size = 4
#     learning_rate = 1e-3
#     d_model = max_len_in

#     config = MambaConfig(d_model=d_model, n_layers=2)
#     model = MambaWithProj(config, binary_classification).to(device)
#     embedding = nn.Embedding(5, d_model).to(device)  # 5 because we have 5 tokens (A, T, C, G, P)
    
#     print(model)
    
#     optim = torch.optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=50, verbose=True)

#     print(f"\n> Training for {epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
        
#         for _ in range(len(data_inputs) // batch_size):
#             input, output, indices = get_batch(data_inputs, data_labels, batch_size)
#             input = embedding(input.to(device))
#             output = output.to(device)

#             optim.zero_grad()
#             logits = model(input)

#             if binary_classification:
#                 loss = F.binary_cross_entropy_with_logits(logits, output.float())
#             else:
#                 loss = F.cross_entropy(logits.view(-1, 5), output.view(-1))

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optim.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / (len(data_inputs) // batch_size)
#         scheduler.step(avg_loss)

#         if (epoch + 1) % 100 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
#             model.eval()
#             with torch.no_grad():
#                 input, output, indices = get_batch(data_inputs, data_labels, batch_size)
#                 input = embedding(input.to(device))
#                 logits = model(input)
                
#                 if binary_classification:
#                     preds = (torch.sigmoid(logits) > 0.5).int().squeeze()
#                     for i in range(batch_size):
#                         print(f"Sample {i+1}:")
#                         print(f"  Input: {data_inputs[indices[i]]}")
#                         print(f"  Prediction: {preds[i].item()}")
#                         print(f"  Actual: {data_labels[indices[i]]}")
#                         print()
#                 else:
#                     preds = torch.argmax(logits, dim=-1)
#                     for i in range(batch_size):
#                         print(f"Sample {i+1}:")
#                         print(f"  Input: {data_inputs[indices[i]]}")
#                         print(f"  Prediction: {''.join(detokenizer(j) for j in preds[i])}")
#                         print(f"  Actual: {data_labels[indices[i]]}")
#                         print()

# if __name__ == "__main__":
#     train()
import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using {device} device")

binary_classification = True  # Set this to False if you're doing sequence-to-sequence prediction

def tokenizer(inputs):
    token = {"A": 0, "T": 1, "C": 2, "G": 3, "P": 4}
    if isinstance(inputs[0], str):
        return [[token[j] for j in i.upper()] for i in inputs]
    return inputs  # For binary labels, return as is

def detokenizer(token):
    detoken = {0: "A", 1: "T", 2: "C", 3: "G", 4: "P"}
    if isinstance(token, torch.Tensor):
        return detoken[token.item()]
    return detoken[token]

class Dataset:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train_inputs = []
        self.train_labels = []
        self.test_inputs = []
        self.test_labels = []
   
    def read_data(self):
        def process_file(file_path, inputs, labels):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            for line in lines:
                input_seq, label = line.strip().split('\t')
                if len(input_seq) < 15000:
                    inputs.append(input_seq)
                    labels.append(int(label) if binary_classification else label)
        
        process_file(self.train_file_path, self.train_inputs, self.train_labels)
        process_file(self.test_file_path, self.test_inputs, self.test_labels)

    def pad_sequences(self):
        all_inputs = self.train_inputs + self.test_inputs
        max_len_in = max(len(i) for i in all_inputs)
        self.train_inputs = [i + "P" * (max_len_in - len(i)) for i in self.train_inputs]
        self.test_inputs = [i + "P" * (max_len_in - len(i)) for i in self.test_inputs]

def get_batch(data_inputs, data_labels, batch_size):
    indices = torch.randperm(len(data_inputs))[:batch_size]
    input_batch = [data_inputs[i] for i in indices]
    label_batch = [data_labels[i] for i in indices]
    return torch.tensor(tokenizer(input_batch)), torch.tensor(label_batch), indices

class MambaWithProj(nn.Module):
    def __init__(self, config, binary_classification):
        super().__init__()
        self.mamba = Mamba(config)
        self.binary_classification = binary_classification
        if binary_classification:
            self.proj = nn.Linear(config.d_model, 1)
        else:
            self.proj = nn.Linear(config.d_model, 5)  # 5 classes: A, T, C, G, P
   
    def forward(self, x):
        x = self.mamba(x)
        if self.binary_classification:
            # Global average pooling
            x = x.mean(dim=1)
        return self.proj(x)
def train():
    # Load and preprocess data
    dataset = Dataset(train_file_path="train.txt", test_file_path="test.txt")
    dataset.read_data()
    dataset.pad_sequences()
   
    train_inputs = tokenizer(dataset.train_inputs)
    train_labels = dataset.train_labels
    test_inputs = tokenizer(dataset.test_inputs)
    test_labels = dataset.test_labels
   
    print(f"Total number of training samples: {len(train_inputs)}")
    print(f"Total number of test samples: {len(test_inputs)}")

    max_len_in = len(train_inputs[0])
   
    epochs = 201
    batch_size = 16  # Increased batch size
    learning_rate = 1e-3
    d_model = 64  # Adjust this value based on your needs
    config = MambaConfig(d_model=d_model, n_layers=2)
    model = MambaWithProj(config, binary_classification).to(device)
    embedding = nn.Embedding(5, d_model).to(device)  # 5 because we have 5 tokens (A, T, C, G, P)
   
    print(model)
   
    optim = torch.optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True)
    print(f"\n> Training for {epochs} epochs with batch size {batch_size} and learning rate {learning_rate}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
       
        for _ in range(len(train_inputs) // batch_size):
            input, output, indices = get_batch(train_inputs, train_labels, batch_size)
            input = embedding(input.to(device))
            output = output.to(device)
            optim.zero_grad()
            logits = model(input)
            if binary_classification:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(), output.float())
            else:
                loss = F.cross_entropy(logits.view(-1, 5), output.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(train_inputs) // batch_size)
        scheduler.step(avg_loss)

        if (epoch + 1) % 1 == 0:  # Increased frequency of logging
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Evaluate on train set
            train_accuracy = evaluate(model, embedding, train_inputs, train_labels, batch_size, "Train")
            
            # Evaluate on test set
            test_accuracy = evaluate(model, embedding, test_inputs, test_labels, batch_size, "Test")
            
            print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

def evaluate(model, embedding, data_inputs, data_labels, batch_size, set_name):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(data_inputs), batch_size):
            batch_inputs = data_inputs[i:i+batch_size]
            batch_labels = data_labels[i:i+batch_size]
            
            input = torch.tensor(batch_inputs).to(device)
            input = embedding(input)
            output = torch.tensor(batch_labels).to(device)
            
            logits = model(input)
            
            if binary_classification:
                preds = (torch.sigmoid(logits) > 0.5).int().squeeze()
                correct += (preds == output).float().sum().item()
            else:
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == output).float().sum().item()
            
            total += len(batch_labels)
    
    accuracy = correct / total
    
    # Print some example predictions
    if binary_classification:
        for i in range(min(5, len(batch_inputs))):
            print(f"{set_name} Sample {i+1}:")
            print(f"  Prediction: {preds[i].item()}")
            print(f"  Actual: {batch_labels[i]}")
            print()
    else:
        for i in range(min(5, len(batch_inputs))):
            print(f"{set_name} Sample {i+1}:")
            print(f"  Input: {''.join(detokenizer(j) for j in batch_inputs[i])[:50]}...")  # First 50 characters
            print(f"  Prediction: {''.join(detokenizer(j) for j in preds[i])[:50]}...")  # First 50 characters
            print(f"  Actual: {batch_labels[i][:50]}...")  # First 50 characters
            print()
    
    return accuracy

if __name__ == "__main__":
    train()