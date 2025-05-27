import torch
import torch.nn as nn
import timm
import librosa
import numpy as np
import torchaudio.transforms as T
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any

class BirdCLEFConfig(PretrainedConfig):
    """
    Configuration class for BirdCLEF model.
    """
    model_type = "birdclef"
    
    def __init__(
        self,
        num_classes: int = 182,
        backbone_name: str = "tf_efficientnetv2_s_in21k",
        drop_rate: float = 0.4,
        drop_path_rate: float = 0.3,
        sample_rate: int = 32000,
        duration: int = 5,
        n_mels: int = 128,
        fmin: int = 20,
        fmax: int = 16000,
        hop_length: int = 512,
        n_fft: int = 2048,
        time_mask: int = 20,
        freq_mask: int = 10,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
        label2idx: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        
        # Audio processing parameters
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Augmentation parameters
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        
        # Label mapping
        self.label2idx = label2idx or {}

class AudioProcessor:
    """Audio processing utilities matching training pipeline"""
    
    def __init__(self, config: BirdCLEFConfig):
        self.config = config
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """Load and standardize audio"""
        try:
            y, sr = librosa.load(filepath, sr=self.config.sample_rate, mono=True)
            y = librosa.util.fix_length(y, size=self.config.duration * self.config.sample_rate)
            return y
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return np.zeros(self.config.duration * self.config.sample_rate)
    
    def create_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Create normalized mel spectrogram identical to training"""
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft
        )
        S = librosa.power_to_db(S, ref=np.max)
        # Normalize to [-1, 1] range (CRITICAL for consistency)
        S = (S - S.min()) / (S.max() - S.min()) * 2 - 1
        return S

class SpecAugment:
    """Spectral augmentation for TTA"""
    
    def __init__(self, config: BirdCLEFConfig):
        self.config = config
        self.time_mask = T.TimeMasking(time_mask_param=config.time_mask)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=config.freq_mask)
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        for _ in range(self.config.n_time_masks):
            spec = self.time_mask(spec)
        for _ in range(self.config.n_freq_masks):
            spec = self.freq_mask(spec)
        return spec

class BirdCLEFModel(PreTrainedModel):
    """
    BirdCLEF model for bird species classification from audio spectrograms.
    
    This model uses EfficientNetV2 as backbone with spectral attention mechanism.
    """
    config_class = BirdCLEFConfig
    
    def __init__(self, config: BirdCLEFConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize audio processor and augmenter
        self.audio_processor = AudioProcessor(config)
        self.spec_augmenter = SpecAugment(config)
        
        # Backbone model
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=True,
            in_chans=1,
            num_classes=config.num_classes,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate
        )
        
        # Spectral attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Store label mappings
        self.label2idx = config.label2idx
        self.idx2label = {v: k for k, v in self.label2idx.items()}
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Tensor of shape (batch_size, 1, n_mels, time_steps)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Apply spectral attention
        attn = self.attention(inputs)
        x = inputs * attn
        
        # Pass through backbone
        return self.backbone(x)
    
    def predict_from_audio(self, audio_path: str, use_tta: bool = True, tta_steps: int = 3) -> Dict[str, Any]:
        """
        Predict bird species from audio file.
        
        Args:
            audio_path: Path to audio file
            use_tta: Whether to use test-time augmentation
            tta_steps: Number of TTA steps
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        self.eval()
        
        # Load and process audio
        y = self.audio_processor.load_audio(audio_path)
        spec = self.audio_processor.create_mel_spectrogram(y)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        device = next(self.parameters()).device
        spec_tensor = spec_tensor.to(device)
        
        with torch.no_grad():
            if use_tta:
                # Test-time augmentation
                outputs = []
                for _ in range(tta_steps):
                    aug_spec = self.spec_augmenter(spec_tensor.clone())
                    output = self(aug_spec)
                    outputs.append(output)
                logits = torch.stack(outputs).mean(0)
            else:
                logits = self(spec_tensor)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], k=5)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                species = self.idx2label.get(idx, f"unknown_{idx}")
                predictions.append({
                    'species': species,
                    'probability': float(prob),
                    'confidence': float(prob)
                })
        
        return {
            'predictions': predictions,
            'top_prediction': predictions[0]['species'],
            'confidence': predictions[0]['confidence']
        }
    
    def predict_from_spectrogram(self, spectrogram: np.ndarray, use_tta: bool = True, tta_steps: int = 3) -> Dict[str, Any]:
        """
        Predict from preprocessed spectrogram.
        
        Args:
            spectrogram: Numpy array of shape (n_mels, time_steps)
            use_tta: Whether to use test-time augmentation
            tta_steps: Number of TTA steps
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        self.eval()
        
        # Convert to tensor
        spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)
        device = next(self.parameters()).device
        spec_tensor = spec_tensor.to(device)
        
        with torch.no_grad():
            if use_tta:
                # Test-time augmentation
                outputs = []
                for _ in range(tta_steps):
                    aug_spec = self.spec_augmenter(spec_tensor.clone())
                    output = self(aug_spec)
                    outputs.append(output)
                logits = torch.stack(outputs).mean(0)
            else:
                logits = self(spec_tensor)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], k=5)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                species = self.idx2label.get(idx, f"unknown_{idx}")
                predictions.append({
                    'species': species,
                    'probability': float(prob),
                    'confidence': float(prob)
                })
        
        return {
            'predictions': predictions,
            'top_prediction': predictions[0]['species'],
            'confidence': predictions[0]['confidence']
        }

# Register the model
BirdCLEFModel.register_for_auto_class("AutoModel")
BirdCLEFConfig.register_for_auto_class("AutoConfig")
