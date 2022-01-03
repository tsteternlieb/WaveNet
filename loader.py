import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Callable


class AudioLoaderFixed(Dataset):
    def __init__(
        self,
        audio_path: str,
        clip_length: int,
        transformations: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        """Dataset for fixed length audio samples

        Args:
            audio_path (str): path to file
            clip_length (int): length of sample
            transformations (Callable[[torch.Tensor], torch.Tensor]): augmentation transformations
        """
        self.audio_path = audio_path
        self.clip_length = clip_length

        info = torchaudio.info(audio_path)
        self.audio_length = info.num_frames
        self.num_channels = info.num_channels
        self.transformations = transformations

    def __len__(self):
        return self.audio_length - self.clip_length

    def __getitem__(self, idx):

        t = torchaudio.load(
            self.audio_path, frame_offset=idx, num_frames=self.clip_length
        )[0]
        if self.num_channels > 1:
            t = t[0]
        if self.transformations:
            t = self.transformations(t)
        return t


def transforms(a):
    pass
