import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from music21 import chord, instrument, note, stream

from model import MusicLSTM


def load_checkpoint(models_dir: Path):
    checkpoint = torch.load(models_dir / 'best_model.pt', map_location='cpu')
    return checkpoint


def sample_next_token(logits, temperature: float = 1.0):
    temperature = max(0.1, temperature)
    probs = torch.softmax(logits / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return idx


def generate_tokens(model, seed_sequence, idx2token, steps: int, temperature: float, device):
    generated = list(seed_sequence)
    hidden = None
    model.eval()
    with torch.no_grad():
        current_sequence = list(seed_sequence)
        for _ in range(steps):
            x = torch.tensor([current_sequence], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            next_idx = sample_next_token(logits[:, -1, :].squeeze(0), temperature=temperature)
            generated.append(next_idx)
            current_sequence = current_sequence[1:] + [next_idx]
    return [idx2token[idx] for idx in generated]


def tokens_to_midi(tokens, output_path: Path):
    output_notes = []
    offset = 0

    for token in tokens:
        if '.' in token or token.isdigit():
            notes_in_chord = token.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note)) if current_note.isdigit() else note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(token)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Generate new music from a trained model.')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--output-path', type=str, default='outputs/generated.mid')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(data_dir / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    token2idx = vocab['token2idx']
    idx2token = vocab['idx2token']

    X = np.load(data_dir / 'X.npy')
    if len(X) == 0:
        raise ValueError('No sequences found. Run preprocess.py first.')

    checkpoint = load_checkpoint(models_dir)
    config = checkpoint['config']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicLSTM(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    seed_sequence = random.choice(X).tolist()
    generated_tokens = generate_tokens(model, seed_sequence, idx2token, args.steps, args.temperature, device)
    tokens_to_midi(generated_tokens, output_path)

    print(f'Generated MIDI saved to: {output_path.resolve()}')


if __name__ == '__main__':
    main()
