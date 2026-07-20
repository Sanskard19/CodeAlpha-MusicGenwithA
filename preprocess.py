import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from music21 import chord, converter, instrument, note
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {'.mid', '.midi'}


def find_midi_files(data_dir: Path) -> List[Path]:
    return sorted(
        [p for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def extract_tokens(midi_path: Path) -> List[str]:
    score = converter.parse(str(midi_path))
    parts = instrument.partitionByInstrument(score)
    elements = parts.parts[0].recurse() if parts else score.flat.notes

    tokens = []
    for element in elements:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append('.'.join(str(n) for n in element.normalOrder))
    return tokens


def build_sequences(encoded_tokens: List[int], seq_len: int):
    inputs = []
    targets = []
    for i in range(0, len(encoded_tokens) - seq_len):
        inputs.append(encoded_tokens[i:i + seq_len])
        targets.append(encoded_tokens[i + seq_len])
    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIDI files into token sequences.')
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--out-dir', type=str, default='data/processed')
    parser.add_argument('--seq-len', type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = find_midi_files(data_dir)
    if not midi_files:
        raise FileNotFoundError('No MIDI files found in data/raw. Add dataset files first.')

    all_tokens = []
    file_token_counts = {}

    for midi_path in tqdm(midi_files, desc='Parsing MIDI files'):
        try:
            tokens = extract_tokens(midi_path)
            if tokens:
                all_tokens.extend(tokens)
                file_token_counts[midi_path.name] = len(tokens)
        except Exception as exc:
            print(f'[WARN] Skipping {midi_path.name}: {exc}')

    if len(all_tokens) <= args.seq_len:
        raise ValueError('Not enough extracted tokens to build training sequences.')

    vocab = sorted(set(all_tokens))
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for token, idx in token2idx.items()}

    encoded = [token2idx[token] for token in all_tokens]
    X, y = build_sequences(encoded, args.seq_len)

    np.save(out_dir / 'X.npy', X)
    np.save(out_dir / 'y.npy', y)

    with open(out_dir / 'vocab.pkl', 'wb') as f:
        pickle.dump({'token2idx': token2idx, 'idx2token': idx2token}, f)

    metadata = {
        'num_midi_files': len(midi_files),
        'total_tokens': len(all_tokens),
        'vocab_size': len(vocab),
        'sequence_length': args.seq_len,
        'num_sequences': int(len(X)),
        'sample_file_token_counts': dict(list(file_token_counts.items())[:10])
    }
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Saved preprocessed files to:', out_dir.resolve())
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
