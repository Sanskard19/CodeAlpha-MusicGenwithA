# Music Generation with AI

A modular academic project that trains an LSTM-based neural network on MIDI files and generates new piano-style music.

## Features
- Collects local MIDI files for training.
- Preprocesses notes and chords into token sequences using `music21`.
- Trains an LSTM model in PyTorch.
- Generates new note sequences and converts them back to MIDI.
- Includes a Streamlit demo app for quick showcase.

## Folder Structure
```
music-gen-ai-project/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── scripts/
│   └── download_or_prepare_dataset.py
├── generate.py
├── model.py
├── preprocess.py
├── train.py
├── requirements.txt
└── README.md
```

## Setup
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put your `.mid` or `.midi` files into `data/raw/`.

## Run Order
### 1) Inspect dataset
```bash
python scripts/download_or_prepare_dataset.py
```

### 2) Preprocess MIDI files
```bash
python preprocess.py --seq-len 64
```

### 3) Train model
```bash
python train.py --epochs 10 --batch-size 64
```

### 4) Generate music
```bash
python generate.py --steps 200 --temperature 1.0
```

Generated MIDI will be saved in `outputs/generated.mid`.

### 5) Run demo app
```bash
streamlit run app/app.py
```

## Notes
- Start with piano/classical MIDI for cleaner results.
- Add dataset source and license info in `data/README.md` before submitting.
- Optional audio conversion can be done later using FluidSynth or MuseScore.
