import os  
import sys


def create_fasta_format(path, output_name='/data/nilar/operon/dataset_creation_analysis/sequence_binary.fasta'):
    with open(path, 'r') as f:
        with open(output_name, 'w') as fasta:
            for i, line in enumerate(f):
                fasta.write(f">ID: {i} | label: {line.split('\t')[1]}")
                fasta.write(f"{line.split('\t')[0]}\n")

# create_fasta_format('/data/nilar/operon/dataset_creation_analysis/sequences_binary.txt')


# def cdhit(path, output_name='/data/nilar/operon/dataset_creation_analysis/cd_hit_cluster.txt'):
#     os.system(f"cd-hit -i {path} -o {output_name} -c 0.85 -M 0")

# cdhit('/data/nilar/operon/dataset_creation_analysis/sequence_binary.fasta')

from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

def parse_fasta(file_path):
    sequences = []
    labels = []
    current_label = None
    current_sequence = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_label is not None:
                    sequences.append(''.join(current_sequence))
                    labels.append(current_label)
                    current_sequence = []
                current_label = int(line.split('label:')[1].strip())
            else:
                current_sequence.append(line)
    if current_label is not None:
        sequences.append(''.join(current_sequence))
        labels.append(current_label)
    return sequences, labels

def write_output(sequences, labels, file_path):
    with open(file_path, 'w') as file:
        for seq, label in zip(sequences, labels):
            file.write(f"{seq}\t{label}\n")

def get_length_bin(length, bin_size=15):
    return length // bin_size

def balanced_stratified_split(file_path, test_size=0.2, bin_size=200):
    sequences, labels = parse_fasta(file_path)
    
    # Group sequences by length bin and label
    grouped_sequences = defaultdict(list)
    for seq, label in zip(sequences, labels):
        length_bin = get_length_bin(len(seq), bin_size)
        grouped_sequences[(length_bin, label)].append(seq)
    
    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []
    
    for (length_bin, label), seqs in grouped_sequences.items():
        if len(seqs) < 2:
            # If there's only one sequence in the bin, add it to the larger set
            if len(train_sequences) <= len(test_sequences):
                train_sequences.extend(seqs)
                train_labels.extend([label] * len(seqs))
            else:
                test_sequences.extend(seqs)
                test_labels.extend([label] * len(seqs))
        else:
            # Split the sequences in this bin
            seq_train, seq_test = train_test_split(seqs, test_size=test_size)
            train_sequences.extend(seq_train)
            train_labels.extend([label] * len(seq_train))
            test_sequences.extend(seq_test)
            test_labels.extend([label] * len(seq_test))
    
    # Write train and test sets to new files
    write_output(train_sequences, train_labels, "train.txt")
    write_output(test_sequences, test_labels, "test.txt")
    
    print(f"Train set: {len(train_sequences)} sequences")
    print(f"Test set: {len(test_sequences)} sequences")
    
    # Calculate and print length statistics
    train_lengths = [len(seq) for seq in train_sequences]
    test_lengths = [len(seq) for seq in test_sequences]
    print(f"\nTrain set length stats: min={min(train_lengths)}, max={max(train_lengths)}, mean={np.mean(train_lengths):.2f}, median={np.median(train_lengths)}")
    print(f"Test set length stats: min={min(test_lengths)}, max={max(test_lengths)}, mean={np.mean(test_lengths):.2f}, median={np.median(test_lengths)}")

if __name__ == "__main__":
    input_file = "/data/nilar/operon/dataset_creation_analysis/cd_hit_cluster.txt"
    balanced_stratified_split(input_file)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import Counter

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    inputs = []
    labels = []
    for line in lines:
        input_seq, label = line.strip().split('\t')
        inputs.append(input_seq)
        labels.append(label)
    
    return inputs, labels

def plot_length_histogram(train_lengths, test_lengths, filename):
    plt.figure(figsize=(12, 6))
    plt.hist(train_lengths, bins=120, alpha=0.7, label='Train', density=True, )
    plt.hist(test_lengths, bins=120, alpha=0.7, label='Test', density=True)
    plt.title('Histogram of Input Sequence Lengths (Normalized)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(-50, 30000)
    plt.savefig(filename)
    plt.close()

def plot_label_distribution(train_labels, test_labels, filename):
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    labels = sorted(set(train_counts.keys()) | set(test_counts.keys()))
    train_values = [train_counts.get(label, 0) / len(train_labels) for label in labels]
    test_values = [test_counts.get(label, 0) / len(test_labels) for label in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_values, width, label='Train', alpha=0.7)
    ax.bar(x + width/2, test_values, width, label='Test', alpha=0.7)
    
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Normalized Class/Label Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.savefig(filename)
    plt.close()

def plot_nucleotide_frequency(train_inputs, test_inputs, filename):
    def count_nucleotides(inputs):
        all_nucleotides = ''.join(inputs)
        return Counter(all_nucleotides)
    
    train_freq = count_nucleotides(train_inputs)
    test_freq = count_nucleotides(test_inputs)
    
    nucleotides = sorted(set(train_freq.keys()) | set(test_freq.keys()))
    train_values = [train_freq.get(nuc, 0) / sum(train_freq.values()) for nuc in nucleotides]
    test_values = [test_freq.get(nuc, 0) / sum(test_freq.values()) for nuc in nucleotides]
    
    x = np.arange(len(nucleotides))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_values, width, label='Train', alpha=0.7)
    ax.bar(x + width/2, test_values, width, label='Test', alpha=0.7)
    
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Normalized Nucleotide Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(nucleotides)
    ax.legend()
    
    plt.savefig(filename)
    plt.close()

def analyze_dna_data(train_file, test_file):
    train_inputs, train_labels = read_data(train_file)
    test_inputs, test_labels = read_data(test_file)
    
    train_lengths = [len(seq) for seq in train_inputs]
    test_lengths = [len(seq) for seq in test_inputs]
    
    plot_length_histogram(train_lengths, test_lengths, 'input_length_histogram.png')
    plot_label_distribution(train_labels, test_labels, 'label_distribution.png')
    plot_nucleotide_frequency(train_inputs, test_inputs, 'nucleotide_frequency.png')
    
    # Print summary statistics
    for dataset, inputs, labels, lengths in [("Train", train_inputs, train_labels, train_lengths),
                                             ("Test", test_inputs, test_labels, test_lengths)]:
        print(f"\n{dataset} Set Statistics:")
        print(f"Total sequences: {len(inputs)}")
        print(f"Number of unique labels: {len(set(labels))}")
        print(f"Average sequence length: {sum(lengths) / len(lengths):.2f}")
        print(f"Minimum sequence length: {min(lengths)}")
        print(f"Maximum sequence length: {max(lengths)}")
        
        print("\nLabel distribution:")
        label_counts = Counter(labels)
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(labels):.2%})")
        
        print("\nNucleotide frequency:")
        nuc_freq = Counter(''.join(inputs))
        total_nucs = sum(nuc_freq.values())
        for nuc, count in nuc_freq.items():
            print(f"  {nuc}: {count} ({count/total_nucs:.2%})")


train_file = 'train.txt'
test_file = 'test.txt'
analyze_dna_data(train_file, test_file)