import argparse
    from pathlib import Path
    import statistics
    from typing import List

    from music21 import converter


    SUPPORTED_EXTENSIONS = {'.mid', '.midi'}


    def find_midi_files(data_dir: Path) -> List[Path]:
        midi_files = []
        for path in data_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                midi_files.append(path)
        return sorted(midi_files)


    def safe_duration_quarters(midi_path: Path):
        try:
            score = converter.parse(str(midi_path))
            return float(score.duration.quarterLength)
        except Exception:
            return None


    def main():
        parser = argparse.ArgumentParser(description='Scan local MIDI files and print basic dataset stats.')
        parser.add_argument('--data-dir', type=str, default='data/raw', help='Folder containing MIDI files.')
        parser.add_argument('--sample', type=int, default=25, help='How many files to sample for duration stats.')
        args = parser.parse_args()

        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        midi_files = find_midi_files(data_dir)
        print(f'Dataset folder: {data_dir.resolve()}')
        print(f'MIDI files found: {len(midi_files)}')

        if not midi_files:
            print('No MIDI files found. Add .mid or .midi files to data/raw and run again.')
            return

        sample_files = midi_files[:max(1, min(args.sample, len(midi_files)))]
        durations = [safe_duration_quarters(path) for path in sample_files]
        valid_durations = [d for d in durations if d is not None]

        print('
Sample files:')
        for path in sample_files[:10]:
            print(f'  - {path.name}')

        if valid_durations:
            print('
Approx duration stats (quarter lengths, sampled files only):')
            print(f'  min   : {min(valid_durations):.2f}')
            print(f'  mean  : {statistics.mean(valid_durations):.2f}')
            print(f'  median: {statistics.median(valid_durations):.2f}')
            print(f'  max   : {max(valid_durations):.2f}')
        else:
            print('
Could not parse sample files for duration stats, but file discovery worked.')


    if __name__ == '__main__':
        main()
