"""
Lip Sync Service

This module contains the core lip sync processing logic.
It provides an abstraction layer for different lip sync libraries.
"""

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Wav2Lip model directory (relative to backend)
WAV2LIP_DIR = Path(__file__).parent.parent.parent / "models" / "wav2lip"

# SadTalker directory (relative to project root)
SADTALKER_DIR = Path(__file__).parent.parent.parent.parent / "models" / "sadtalker"


@dataclass
class LipSyncStatus:
    """Status tracking for lip sync jobs"""
    job_id: str
    status: str  # queued, processing, completed, failed, cancelled
    progress: float = 0.0
    output_url: str | None = None
    error: str | None = None


@dataclass
class LipSyncConfig:
    """Configuration for lip sync processing"""
    model: str = "wav2lip"
    quality: str = "medium"
    fps: int = 25
    resize_factor: int = 1
    pads: list[int] = field(default_factory=lambda: [0, 10, 0, 0])
    face_det_batch_size: int = 16
    wav2lip_batch_size: int = 128
    nosmooth: bool = False
    extra_options: dict[str, Any] = field(default_factory=dict)


class LipSyncModel(Protocol):
    """Protocol for lip sync model implementations"""

    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        config: LipSyncConfig
    ) -> bool:
        """Process video/image with audio to generate lip-synced output"""
        ...


class AudioProcessor:
    """Audio processing for Wav2Lip using librosa"""

    # Hyperparameters matching Wav2Lip
    num_mels = 80
    n_fft = 800
    hop_size = 200
    win_size = 800
    sample_rate = 16000
    preemphasis = 0.97
    preemphasize = True
    signal_normalization = True
    allow_clipping_in_normalization = True
    symmetric_mels = True
    max_abs_value = 4.0
    min_level_db = -100
    ref_level_db = 20
    fmin = 55
    fmax = 7600

    _mel_basis = None

    @classmethod
    def load_wav(cls, path: str) -> np.ndarray:
        """Load audio file and convert to 16kHz"""
        import librosa
        wav, _ = librosa.load(path, sr=cls.sample_rate)
        return wav

    @classmethod
    def _preemphasis(cls, wav: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter"""
        from scipy import signal
        if cls.preemphasize:
            return signal.lfilter([1, -cls.preemphasis], [1], wav)
        return wav

    @classmethod
    def _stft(cls, y: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform"""
        import librosa
        return librosa.stft(
            y=y,
            n_fft=cls.n_fft,
            hop_length=cls.hop_size,
            win_length=cls.win_size
        )

    @classmethod
    def _build_mel_basis(cls) -> np.ndarray:
        """Build mel filterbank"""
        import librosa
        return librosa.filters.mel(
            sr=cls.sample_rate,
            n_fft=cls.n_fft,
            n_mels=cls.num_mels,
            fmin=cls.fmin,
            fmax=cls.fmax
        )

    @classmethod
    def _linear_to_mel(cls, spectrogram: np.ndarray) -> np.ndarray:
        """Convert linear spectrogram to mel spectrogram"""
        if cls._mel_basis is None:
            cls._mel_basis = cls._build_mel_basis()
        return np.dot(cls._mel_basis, spectrogram)

    @classmethod
    def _amp_to_db(cls, x: np.ndarray) -> np.ndarray:
        """Convert amplitude to decibels"""
        min_level = np.exp(cls.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @classmethod
    def _normalize(cls, S: np.ndarray) -> np.ndarray:
        """Normalize spectrogram"""
        if cls.allow_clipping_in_normalization:
            if cls.symmetric_mels:
                return np.clip(
                    (2 * cls.max_abs_value) * ((S - cls.min_level_db) / (-cls.min_level_db)) - cls.max_abs_value,
                    -cls.max_abs_value,
                    cls.max_abs_value
                )
            else:
                return np.clip(
                    cls.max_abs_value * ((S - cls.min_level_db) / (-cls.min_level_db)),
                    0,
                    cls.max_abs_value
                )
        if cls.symmetric_mels:
            return (2 * cls.max_abs_value) * ((S - cls.min_level_db) / (-cls.min_level_db)) - cls.max_abs_value
        return cls.max_abs_value * ((S - cls.min_level_db) / (-cls.min_level_db))

    @classmethod
    def melspectrogram(cls, wav: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from waveform"""
        D = cls._stft(cls._preemphasis(wav))
        S = cls._amp_to_db(cls._linear_to_mel(np.abs(D))) - cls.ref_level_db
        if cls.signal_normalization:
            return cls._normalize(S)
        return S


class Wav2LipProcessor:
    """
    Wav2Lip implementation for lip sync.
    """

    mel_step_size = 16
    img_size = 96

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.face_detector = None
        logger.info(f"Wav2Lip using device: {self.device}")

    def _load_wav2lip_model(self, checkpoint_path: str) -> Any:
        """Load Wav2Lip model from checkpoint"""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Try loading as TorchScript model first
        try:
            if self.device == "cuda":
                model = torch.jit.load(checkpoint_path)
            else:
                model = torch.jit.load(checkpoint_path, map_location=torch.device("cpu"))
            logger.info("Loaded as TorchScript model")
            model = model.to(self.device)
            return model.eval()
        except Exception as e:
            logger.info(f"Not a TorchScript model, trying state_dict: {e}")

        # Fallback to state_dict loading
        wav2lip_path = str(WAV2LIP_DIR)
        if wav2lip_path not in sys.path:
            sys.path.insert(0, wav2lip_path)

        from models import Wav2Lip

        model = Wav2Lip()

        if self.device == "cuda":
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        else:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False
            )

        # Handle 'module.' prefix from DataParallel
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        return model.eval()

    def _init_face_detector(self) -> Any:
        """Initialize face detector"""
        wav2lip_path = str(WAV2LIP_DIR)
        if wav2lip_path not in sys.path:
            sys.path.insert(0, wav2lip_path)

        import face_detection
        return face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device=self.device
        )

    def _get_smoothened_boxes(self, boxes: np.ndarray, T: int = 5) -> np.ndarray:
        """Smooth face detection boxes over time"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def _face_detect(
        self,
        images: list[np.ndarray],
        pads: list[int],
        batch_size: int,
        nosmooth: bool
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """Detect faces in images"""
        if self.face_detector is None:
            self.face_detector = self._init_face_detector()

        predictions = []
        while True:
            try:
                for i in range(0, len(images), batch_size):
                    batch = np.array(images[i:i + batch_size])
                    preds = self.face_detector.get_detections_for_batch(batch)
                    predictions.extend(preds)
                break
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError("Image too big for face detection")
                batch_size //= 2
                logger.warning(f"OOM in face detection, reducing batch to {batch_size}")

        pady1, pady2, padx1, padx2 = pads
        results = []

        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError("Face not detected in frame")

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth:
            boxes = self._get_smoothened_boxes(boxes, T=5)

        return [
            (image[y1:y2, x1:x2], (y1, y2, x1, x2))
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

    def _datagen(
        self,
        frames: list[np.ndarray],
        mels: list[np.ndarray],
        face_det_results: list[tuple[np.ndarray, tuple]],
        batch_size: int,
        static: bool = False
    ):
        """Generate batches for inference"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i, m in enumerate(mels):
            idx = 0 if static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx]
            face = face.copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= batch_size:
                img_batch_arr = np.asarray(img_batch)
                mel_batch_arr = np.asarray(mel_batch)

                img_masked = img_batch_arr.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch_arr = np.concatenate((img_masked, img_batch_arr), axis=3) / 255.0
                mel_batch_arr = np.reshape(
                    mel_batch_arr,
                    [len(mel_batch_arr), mel_batch_arr.shape[1], mel_batch_arr.shape[2], 1]
                )

                yield img_batch_arr, mel_batch_arr, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch_arr = np.asarray(img_batch)
            mel_batch_arr = np.asarray(mel_batch)

            img_masked = img_batch_arr.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch_arr = np.concatenate((img_masked, img_batch_arr), axis=3) / 255.0
            mel_batch_arr = np.reshape(
                mel_batch_arr,
                [len(mel_batch_arr), mel_batch_arr.shape[1], mel_batch_arr.shape[2], 1]
            )

            yield img_batch_arr, mel_batch_arr, frame_batch, coords_batch

    def _extract_audio(self, audio_path: str, output_path: str) -> str:
        """Convert audio to WAV format if needed"""
        if audio_path.endswith(".wav"):
            return audio_path

        cmd = ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _read_frames(
        self,
        video_path: str,
        resize_factor: int = 1
    ) -> tuple[list[np.ndarray], float, bool]:
        """Read frames from video or image"""
        ext = Path(video_path).suffix.lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Could not read image: {video_path}")
            return [frame], 25.0, True

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if resize_factor > 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w // resize_factor, h // resize_factor))

            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"Could not read video: {video_path}")

        return frames, fps, False

    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        config: LipSyncConfig
    ) -> bool:
        """
        Process video/image with Wav2Lip to generate lip-synced output.
        """
        try:
            logger.info(f"Processing: {video_path} + {audio_path} -> {output_path}")

            # Get checkpoint path
            checkpoint_path = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
            if not checkpoint_path.exists():
                # Fallback to wav2lip.pth
                checkpoint_path = WAV2LIP_DIR / "checkpoints" / "wav2lip.pth"

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Wav2Lip checkpoint not found at {checkpoint_path}")

            # Load model if not already loaded
            if self.model is None:
                self.model = self._load_wav2lip_model(str(checkpoint_path))

            # Read video frames
            frames, fps, is_static = self._read_frames(video_path, config.resize_factor)
            logger.info(f"Read {len(frames)} frames at {fps} fps")

            # Process audio
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_audio = Path(temp_dir) / "audio.wav"
                wav_path = self._extract_audio(audio_path, str(temp_audio))

                wav = AudioProcessor.load_wav(wav_path)
                mel = AudioProcessor.melspectrogram(wav)
                logger.info(f"Mel spectrogram shape: {mel.shape}")

                if np.isnan(mel.reshape(-1)).sum() > 0:
                    raise ValueError("Mel spectrogram contains NaN values")

                # Split mel into chunks
                mel_chunks = []
                mel_idx_multiplier = 80.0 / fps
                i = 0
                while True:
                    start_idx = int(i * mel_idx_multiplier)
                    if start_idx + self.mel_step_size > mel.shape[1]:
                        mel_chunks.append(mel[:, mel.shape[1] - self.mel_step_size:])
                        break
                    mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
                    i += 1

                logger.info(f"Generated {len(mel_chunks)} mel chunks")

                # Adjust frames to match audio
                frames = frames[:len(mel_chunks)]

                # Detect faces
                if is_static:
                    face_det_results = self._face_detect(
                        [frames[0]],
                        config.pads,
                        config.face_det_batch_size,
                        config.nosmooth
                    )
                else:
                    face_det_results = self._face_detect(
                        frames,
                        config.pads,
                        config.face_det_batch_size,
                        config.nosmooth
                    )

                # Create temp output video
                temp_video = Path(temp_dir) / "temp_result.avi"
                frame_h, frame_w = frames[0].shape[:2]
                out = cv2.VideoWriter(
                    str(temp_video),
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    fps,
                    (frame_w, frame_h)
                )

                # Process batches
                gen = self._datagen(
                    frames.copy(),
                    mel_chunks,
                    face_det_results,
                    config.wav2lip_batch_size,
                    is_static
                )

                for img_batch, mel_batch, batch_frames, batch_coords in gen:
                    img_batch = torch.FloatTensor(
                        np.transpose(img_batch, (0, 3, 1, 2))
                    ).to(self.device)
                    mel_batch = torch.FloatTensor(
                        np.transpose(mel_batch, (0, 3, 1, 2))
                    ).to(self.device)

                    with torch.no_grad():
                        pred = self.model(mel_batch, img_batch)

                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                    for p, f, c in zip(pred, batch_frames, batch_coords):
                        y1, y2, x1, x2 = c
                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = p
                        out.write(f)

                out.release()

                # Combine with audio using ffmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-i", wav_path,
                    "-i", str(temp_video),
                    "-strict", "-2",
                    "-q:v", "1",
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            logger.info(f"Successfully generated: {output_path}")
            return True

        except Exception as e:
            logger.exception(f"Wav2Lip processing failed: {e}")
            raise


class SadTalkerProcessor:
    """
    SadTalker implementation for talking face generation.
    Uses the SadTalker library for audio-driven talking head generation.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_model = None
        self.audio_to_coeff = None
        self.animate_from_coeff = None
        self.sadtalker_paths = None
        self._initialized = False
        logger.info(f"SadTalker using device: {self.device}")

    def _init_models(self, size: int = 256) -> None:
        """Initialize SadTalker models"""
        if self._initialized:
            return

        sadtalker_path = str(SADTALKER_DIR)
        if sadtalker_path not in sys.path:
            sys.path.insert(0, sadtalker_path)

        from src.utils.preprocess import CropAndExtract
        from src.test_audio2coeff import Audio2Coeff
        from src.facerender.animate import AnimateFromCoeff
        from src.utils.init_path import init_path

        checkpoint_dir = SADTALKER_DIR / "checkpoints"
        config_dir = SADTALKER_DIR / "src" / "config"

        self.sadtalker_paths = init_path(
            str(checkpoint_dir),
            str(config_dir),
            size,
            False,  # old_version
            "crop"  # preprocess
        )

        logger.info("Loading SadTalker preprocess model...")
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)

        logger.info("Loading SadTalker audio to coeff model...")
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)

        logger.info("Loading SadTalker animation model...")
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

        self._initialized = True
        logger.info("SadTalker models initialized successfully")

    def process(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        config: LipSyncConfig
    ) -> bool:
        """
        Generate talking face video from image and audio using SadTalker.

        Args:
            image_path: Path to source image
            audio_path: Path to audio file
            output_path: Path for output video
            config: LipSync configuration

        Returns:
            True if successful
        """
        import shutil
        from time import strftime

        sadtalker_path = str(SADTALKER_DIR)
        if sadtalker_path not in sys.path:
            sys.path.insert(0, sadtalker_path)

        from src.generate_batch import get_data
        from src.generate_facerender_batch import get_facerender_data

        try:
            logger.info(f"Processing with SadTalker: {image_path} + {audio_path}")

            # Get size from config (256 or 512)
            size = 512 if config.quality == "high" else 256

            # Initialize models if needed
            self._init_models(size)

            # Create save directory
            save_dir = Path(output_path).parent / f"sadtalker_{strftime('%Y_%m_%d_%H.%M.%S')}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Extract 3DMM from source image
            first_frame_dir = save_dir / "first_frame_dir"
            first_frame_dir.mkdir(exist_ok=True)

            logger.info("Extracting 3DMM from source image...")
            first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
                str(image_path),
                str(first_frame_dir),
                "crop",  # preprocess mode
                source_image_flag=True,
                pic_size=size
            )

            if first_coeff_path is None:
                raise ValueError("Could not extract face coefficients from source image")

            # Audio to coefficients
            logger.info("Converting audio to coefficients...")
            batch = get_data(
                first_coeff_path,
                str(audio_path),
                self.device,
                ref_eyeblink_coeff_path=None,
                still=True  # Still mode for single image
            )

            coeff_path = self.audio_to_coeff.generate(
                batch,
                str(save_dir),
                pose_style=0,
                ref_pose_coeff_path=None
            )

            # Generate animation
            logger.info("Generating animation...")
            data = get_facerender_data(
                coeff_path,
                crop_pic_path,
                first_coeff_path,
                str(audio_path),
                batch_size=2,
                input_yaw_list=None,
                input_pitch_list=None,
                input_roll_list=None,
                expression_scale=1.0,
                still_mode=True,
                preprocess="crop",
                size=size
            )

            # Use enhancer if high quality is requested
            enhancer = "gfpgan" if config.quality == "high" else None

            result = self.animate_from_coeff.generate(
                data,
                str(save_dir),
                str(image_path),
                crop_info,
                enhancer=enhancer,
                background_enhancer=None,
                preprocess="crop",
                img_size=size
            )

            # Move result to output path
            if result and Path(result).exists():
                shutil.move(result, output_path)
                logger.info(f"Successfully generated: {output_path}")

                # Cleanup temp directory
                try:
                    shutil.rmtree(save_dir)
                except Exception:
                    pass

                return True
            else:
                raise RuntimeError("SadTalker did not produce output")

        except Exception as e:
            logger.exception(f"SadTalker processing failed: {e}")
            raise


class VideoReTalkingProcessor:
    """
    VideoReTalking implementation for video lip sync.
    """

    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        config: LipSyncConfig
    ) -> bool:
        """
        Process video with VideoReTalking.

        This is a placeholder - implement actual VideoReTalking inference here.
        """
        logger.info(f"Processing with VideoReTalking: {video_path} + {audio_path}")
        raise NotImplementedError("VideoReTalking processor not yet implemented")


class LipSyncService:
    """Main service for lip sync operations"""

    _processors: dict[str, type] = {
        "wav2lip": Wav2LipProcessor,
        "sadtalker": SadTalkerProcessor,
        "video_retalking": VideoReTalkingProcessor,
    }

    _processor_instances: dict[str, Any] = {}

    @classmethod
    def get_processor(cls, model_name: str) -> Any:
        """Get the appropriate processor for the model (cached)"""
        if model_name not in cls._processor_instances:
            processor_class = cls._processors.get(model_name)
            if not processor_class:
                raise ValueError(f"Unknown model: {model_name}")
            cls._processor_instances[model_name] = processor_class()
        return cls._processor_instances[model_name]

    @classmethod
    async def process_lipsync(
        cls,
        job_id: str,
        video_path: str,
        audio_path: str,
        config: Any,
        jobs: dict[str, LipSyncStatus]
    ) -> None:
        """
        Background task to process lip sync job.
        """
        try:
            jobs[job_id].status = "processing"
            jobs[job_id].progress = 0.1

            # Get the appropriate processor
            processor = cls.get_processor(config.model)

            # Determine output path
            output_dir = Path(video_path).parent
            output_path = output_dir / "output.mp4"

            jobs[job_id].progress = 0.2

            # Convert config to LipSyncConfig if needed
            lipsync_config = LipSyncConfig(
                model=config.model,
                quality=config.quality,
                fps=getattr(config, "fps", 25),
                resize_factor=getattr(config, "resize_factor", 1),
            )

            jobs[job_id].progress = 0.3

            # Process the lip sync
            # SadTalker uses image_path instead of video_path
            if config.model == "sadtalker":
                success = processor.process(
                    image_path=video_path,
                    audio_path=audio_path,
                    output_path=str(output_path),
                    config=lipsync_config
                )
            else:
                success = processor.process(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=str(output_path),
                    config=lipsync_config
                )

            if success:
                jobs[job_id].status = "completed"
                jobs[job_id].progress = 1.0
                jobs[job_id].output_url = f"/api/v1/lipsync/download/{job_id}"
                logger.info(f"Job {job_id} completed successfully")
            else:
                jobs[job_id].status = "failed"
                jobs[job_id].error = "Processing failed"
                logger.error(f"Job {job_id} failed")

        except Exception as e:
            logger.exception(f"Error processing job {job_id}")
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(e)
