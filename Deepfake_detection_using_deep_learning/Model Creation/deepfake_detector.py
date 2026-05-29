import argparse
from pathlib import Path
import torch

from deepfake_helpers import (
    DeepfakeModel,
    allowed_video_file,
    load_model_weights,
    predict_video,
    predict_video_ensemble,
    select_model_by_sequence_length,
    select_top_models_by_sequence_length,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Standalone deepfake detection script using a ResNeXt + LSTM model.'
    )
    parser.add_argument(
        '--video',
        default=None,
        help='Path to the input video file. If omitted, the latest uploaded video in Django Application/uploaded_videos will be used.',
    )
    parser.add_argument('--model', help='Path to the trained model (.pt file).')
    parser.add_argument(
        '--model-folder',
        default=None,
        help='Folder containing .pt models named with the format model_<acc>_acc_<seq>_frames_final_data.pt. If omitted, the script also searches Model Creation and Django Application/models.',
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Number of frames to sample from the video for prediction.',
    )
    parser.add_argument(
        '--n-clips',
        type=int,
        default=10,
        help='Number of temporal clips to ensemble for prediction (higher can improve stability).',
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Disable using a pretrained backbone. By default a pretrained backbone is used for better accuracy.',
    )
    parser.add_argument(
        '--legacy-transforms',
        action='store_true',
        help='Use legacy preprocessing (Resize to sequence size) to match older model training.',
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run the model on.',
    )
    parser.add_argument(
        '--ensemble-size',
        type=int,
        default=1,
        help='Number of top-trained models to ensemble. If >1, the script searches model folders for top models.',
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.08,
        help='Minimum probability gap required to label a video as FAKE or REAL. Lower values are more permissive.',
    )
    return parser.parse_args()


def resolve_video_path(video_arg):
    default_video_dir = Path(__file__).resolve().parent.parent / 'Django Application' / 'uploaded_videos'
    default_video_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_video_in_folder(folder: Path):
        videos = [p for p in folder.iterdir() if p.is_file() and allowed_video_file(p.name)]
        if not videos:
            return None
        return sorted(videos, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    if video_arg:
        candidate = Path(video_arg)
        if candidate.is_dir():
            latest = find_latest_video_in_folder(candidate)
            if latest:
                print(f'Using latest video from provided folder: {latest}')
                return latest
            raise FileNotFoundError(f'No supported videos found in directory: {candidate}')
        return candidate

    latest_uploaded = find_latest_video_in_folder(default_video_dir)
    if latest_uploaded:
        print(f'No --video provided; using latest uploaded video: {latest_uploaded}')
        return latest_uploaded

    latest_cwd = find_latest_video_in_folder(Path.cwd())
    if latest_cwd:
        print(f'No --video provided; using latest video from current working directory: {latest_cwd}')
        return latest_cwd

    raise FileNotFoundError(
        f'No video provided and no supported video files were found in "{default_video_dir}" or the current working directory. '
        'Please add a supported video file to the upload folder or provide --video <path>.'
    )


def main():
    args = parse_args()
    video_path = resolve_video_path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not allowed_video_file(video_path.name):
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if args.device == 'cuda' and not use_cuda:
        print('CUDA requested but not available; falling back to CPU.')

    def resolve_model_path(model_arg, model_folder_arg, sequence_length):
        if model_arg:
            model_path = Path(model_arg)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return model_path

        candidate_folders = []
        if model_folder_arg:
            candidate_folders.append(Path(model_folder_arg))
        else:
            candidate_folders.append(Path(__file__).resolve().parent.parent / 'Django Application' / 'models')
            candidate_folders.append(Path(__file__).resolve().parent)
            candidate_folders.append(Path.cwd())

        for folder in candidate_folders:
            if not folder.exists():
                continue
            try:
                return Path(select_model_by_sequence_length(str(folder), sequence_length))
            except FileNotFoundError:
                continue

        searched = ', '.join(str(p) for p in candidate_folders)
        raise FileNotFoundError(
            f'No model file found for sequence length {sequence_length} in any searched folder: {searched}. '
            'Provide --model <path> or place a matching .pt file in one of the search directories.'
        )

    model_path = resolve_model_path(args.model, args.model_folder, args.sequence_length)

    print(f"Using model: {model_path}")
    print(f"Using video: {video_path}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Device: {device}")

    # Use pretrained backbone by default unless explicitly disabled
    pretrained_flag = not getattr(args, 'no_pretrained', False)
    # If ensemble requested, find top models and run ensemble prediction
    if args.ensemble_size and args.ensemble_size > 1:
        # Determine candidate folders to search for models
        candidate_folders = []
        if args.model_folder:
            candidate_folders.append(Path(args.model_folder))
        else:
            candidate_folders.append(Path(__file__).resolve().parent.parent / 'Django Application' / 'models')
            candidate_folders.append(Path(__file__).resolve().parent)
            candidate_folders.append(Path.cwd())

        model_paths = []
        for folder in candidate_folders:
            if not folder.exists():
                continue
            try:
                model_paths = select_top_models_by_sequence_length(str(folder), args.sequence_length, top_k=args.ensemble_size)
                if model_paths:
                    break
            except FileNotFoundError:
                continue

        if not model_paths:
            raise FileNotFoundError('Unable to find enough models for ensemble; provide --model or --model-folder with matching .pt files.')

        label, confidence, fake_prob, real_prob = predict_video_ensemble(
            model_paths,
            str(video_path),
            args.sequence_length,
            device,
            threshold=args.confidence_threshold,
            n_clips=args.n_clips,
            use_legacy_transforms=args.legacy_transforms,
            use_tta=True,
            pretrained_backbone=not getattr(args, 'no_pretrained', False),
        )
    else:
        model = DeepfakeModel(num_classes=2, pretrained=pretrained_flag)
        load_model_weights(model, str(model_path), device)

        label, confidence, fake_prob, real_prob = predict_video(
            model,
            str(video_path),
            args.sequence_length,
            device,
            threshold=args.confidence_threshold,
            n_clips=args.n_clips,
            use_legacy_transforms=args.legacy_transforms,
            use_tta=True,
        )

    # Always display only REAL or FAKE (map UNCERTAIN to the higher-probability class)
    # Use the returned label unless it is UNCERTAIN, then pick the higher-probability class
    if label == 'UNCERTAIN':
        final_label = 'REAL' if real_prob > fake_prob else 'FAKE'
    else:
        final_label = label

    # Print model(s) used
    if args.ensemble_size and args.ensemble_size > 1:
        print('Ensembled models:')
        for mp in model_paths:
            print(f' - {mp}')
    else:
        print(f'Using model: {model_path}')

    # Print detailed probabilities and final decision
    print(f'Fake probability: {fake_prob:.2f}%')
    print(f'Real probability: {real_prob:.2f}%')
    print(f'Confidence: {confidence:.2f}%')
    print(f'Final decision: {final_label}')


if __name__ == '__main__':
    main()
