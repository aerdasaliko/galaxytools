import argparse
import csv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

def parse_fasta(fasta_file):
    sequences = []
    header = None
    current_seq = []
    with open(fasta_file, "r") as fasta:
        for line in fasta:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(current_seq)))
                header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if header is not None:
            sequences.append((header, "".join(current_seq)))
    return sequences


def mean_pooling(last_hidden_state, attention_mask):
    """Attention-mask aware mean pooling."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def max_pooling(last_hidden_state, attention_mask):
    """Mask-aware max pooling."""
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_state.masked_fill(mask == 0, float("-inf"))
    return torch.max(masked, dim=1).values


def embed_sequences(sequences, tokenizer, model, device, batch_size=16, pooling="mean"):
    headers, seqs = zip(*sequences)
    embeddings = []

    print("Running batched inference...")
    for i in range(0, len(seqs), batch_size):
        batch = list(seqs[i:i + batch_size])
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(inputs.keys())

        with torch.inference_mode():
            outputs = model(**inputs)[0]
            if pooling == "mean":
                emb = mean_pooling(outputs, inputs["attention_mask"])
            elif pooling == "max":
                emb = max_pooling(outputs, inputs["attention_mask"])
            else:
                raise ValueError("Pooling must be 'mean' or 'max'")
            
        embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return headers, embeddings


def save_to_tsv(headers, embeddings, output_file):
    dim = embeddings.shape[1]
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id"] + [f"dim_{i}" for i in range(dim)])
        for h, emb in zip(headers, embeddings):
            writer.writerow([h] + emb.tolist())


def main():
    parser = argparse.ArgumentParser(description="DNABERT-2 FASTA embedding script")
    parser.add_argument("--model", type=str,
                        default="zhihan1996/DNABERT-2-117M",
                        help="Model name or path")
    parser.add_argument("--fasta", type=str, required=True,
                        help="Input FASTA file")
    parser.add_argument("--output", type=str, default="embeddings.tsv",
                        help="Output TSV file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")

    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    print("Reading FASTA...")
    sequences = parse_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences")

    print("Embedding...")
    headers, embeddings = embed_sequences(
        sequences,
        tokenizer,
        model,
        device=device,
        batch_size=args.batch_size,
        pooling=args.pooling
    )

    print("Saving TSV...")
    save_to_tsv(headers, embeddings, args.output)

    print("Done.")
    print("Output shape:", embeddings.shape)


if __name__ == "__main__":
    main()