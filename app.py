import pickle
import random
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from generate import generate_tokens, tokens_to_midi
from model import MusicLSTM


@st.cache_resource
def load_assets():
    data_dir = ROOT / 'data' / 'processed'
    models_dir = ROOT / 'models'

    with open(data_dir / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    X = np.load(data_dir / 'X.npy')
    checkpoint = torch.load(models_dir / 'best_model.pt', map_location='cpu')
    model = MusicLSTM(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return vocab, X, model


st.set_page_config(page_title='AI Music Generator', layout='centered')
st.title('🎵 AI Music Generator')
st.caption('Generate piano-style MIDI music with an LSTM model')

try:
    vocab, X, model = load_assets()
    idx2token = vocab['idx2token']

    temperature = st.slider('Creativity (temperature)', 0.5, 2.0, 1.0, 0.1)
    steps = st.slider('Number of generated steps', 50, 300, 150, 10)

    if st.button('Generate Music'):
        seed = random.choice(X).tolist()
        generated = generate_tokens(model, seed, idx2token, steps, temperature, torch.device('cpu'))
        out_path = ROOT / 'outputs' / 'streamlit_generated.mid'
        tokens_to_midi(generated, out_path)
        st.success('Music generated successfully.')
        with open(out_path, 'rb') as f:
            st.download_button('Download MIDI', data=f.read(), file_name='generated_music.mid', mime='audio/midi')

    st.info('Run preprocess.py and train.py first before starting the app.')
except Exception as exc:
    st.error(f'App could not load model assets yet: {exc}')
    st.warning('Expected files: data/processed/X.npy, data/processed/vocab.pkl, models/best_model.pt')
