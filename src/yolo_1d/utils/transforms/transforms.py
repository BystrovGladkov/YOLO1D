from typing import Tuple
import numpy as np
import torch
from torch.nn import Module

class Padding(Module):
    
    def __init__(self, length: int):
        """
            Performs a padding of signal to desired length.
        """
        super().__init__()
        self.length = length
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = signals.size(1)
        result = torch.zeros((signals.size(0), self.length), dtype=signals.dtype)
        result[:,:length] = signals[:, :]
        scale = length / self.length
        bboxes *= scale
        return result, bboxes, labels

class Resize(Module):
    
    def __init__(self, length: int):
        """
            Preforms resize of signal to desired length.
        """
        super().__init__()
        self.length = length
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signals = torch.nn.functional.interpolate(signals.unsqueeze(0), self.length, mode='linear').squeeze(0)
        return signals, bboxes, labels
  
   
class RandomResize(Module):
    
    def __init__(self, coef: float):
        """
            Performs random resize of signals using interpolation.
            
            coef
                Coefficient of random resize.
        """
        super().__init__()
        self.coef = coef
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
            Parameters
            -----
            
            signals
                Original signals of shape (channels, length)
            bboxes
                Bounding boxes on image in format (center, width)
            labels 
                Classes of bounding boxes
                
            Returns
            -----
            
            Tuple (signals_transformed, bboxes, labels)
            
            signals_transformed
                Resized signals to random length.
            bboxes
                Without any changes.
            labels
                Without any changes,
        """
        length = signals.size(1) 
        # random resize
        augment_length = int(length * np.random.uniform(1 - self.coef, 1 + self.coef))
        signals = torch.nn.functional.interpolate(signals.unsqueeze(0), augment_length, mode='linear').squeeze(0)
        return signals, bboxes, labels
    
class RandomCrop(Module):
    
    def __init__(self, length: int):
        """
            Performs a random crop of signals to desired length. 
        """
        super().__init__()
        self.length = length
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
            Parameters
            -----
            
            signals
                Original signals of shape (channels, length)
            bboxes
                Bounding boxes on image in format (center, width)
            labels 
                Classes of bounding boxes
                
            Returns
            -----
            
            Tuple (signals_transformed, bboxes, labels)
            
            signals_transformed
                Cropped signals to set length.
            bboxes
                .
            labels
                ,
        """
        length = signals.size(1)
        if length <= self.length:
            # copping is not needed, signal is shorter
            return signals, bboxes, labels
        else:
            # data is larger, need to crop
            pos = np.random.randint(0, length - self.length)
            result = torch.zeros((signals.size(0), self.length))
            # crop signals
            result[:, :] = signals[:, pos: pos + self.length]
            # left and right corners of bboxes
            x1, x2 = bboxes[:, 0] - bboxes[:, 1]/2, bboxes[:, 0] + bboxes[:, 1]/2
            # left and right corners of cropped image
            begin, end = torch.tensor(pos / length), torch.tensor((pos + self.length) / length)
            # crop boxes
            x1 = torch.fmax(x1, begin)
            x2 = torch.fmin(x2, end)
            # mask of boxes that fit the cropped part of image
            mask = x1 < x2
            bboxes = torch.zeros(size=(torch.sum(mask), 2) ,dtype=bboxes.dtype)
            # convert boxes back to (center, width) and scale
            scale = length / self.length
            bboxes[:, 0] = ((x1[mask] + x2[mask])/2 - pos/length) * scale
            bboxes[:, 1] = (x2[mask] - x1[mask]) * scale
            labels = labels[mask]
            return result, bboxes, labels

class RandomPadding(Module):
    def __init__(self, length: int):
        """
            Performs a random padding of signals to desired length. 
        """
        super().__init__()
        self.length = length
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        length = signals.size(1)
        if length >= self.length:
            # padding is not needed
            return signals, bboxes, labels
        else: 
            # data is shorter, need padding
            # generate random position
            pos = np.random.randint(0, self.length - length)
            # tensor with zeros
            result = torch.zeros((signals.size(0), self.length), dtype=signals.dtype)
            result[:, pos:pos+length] = signals
            scale = length / self.length
            # transform center
            bboxes[:, 0] = pos / self.length + bboxes[:, 0] * scale
            # transform width
            bboxes[:, 1] = bboxes[:, 1] * scale
            return result, bboxes, labels   

class RandomReverse(Module):
    
    def __init__(self, p=0.5):
        """
            Performs random reverse in time of signals with probability `p`.
        """
        super().__init__()
        self.p = p

    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if np.random.uniform(0, 1) < self.p:
            bboxes[:,0] = 1 - bboxes[:,0]
            signals = signals.flip(1)
        return signals, bboxes, labels
        
class RandomInvert(Module):
    
    def __init__(self, p = 0.5):
        """
            Randomly inverts values of signals with probability `p`.
        """
        super().__init__()
        self.p = p
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if np.random.uniform(0, 1)<self.p:
            signals = -signals
        return signals, bboxes, labels
    
class RandomJitter(Module):
    
    def __init__(self, sigma: float):
        """
            Adds normal distributed noise N(0,`sigma`^2)
        """
        super().__init__() 
        self.sigma = sigma
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signals += torch.normal(0, self.sigma, signals.size(), dtype=signals.dtype)
        return signals, bboxes, labels

class TimeMasking(Module):

    def __init__(self, max_length = 0.1):
        """
            Sets random range of maximum length `max_length` to zero in signal.
        """
        super().__init__()
        self.max_length = max_length
    
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = signals.size(1)
        # generate size of erased area
        mask_length = np.random.randint(0, int(length*self.max_length))
        # generate position of beginning of erased area
        pos = np.random.randint(0, length - mask_length)
        # set part of signals to zero
        signals[:, pos:pos+mask_length] = 0
        # begin and end of erased elements
        begin, end = torch.tensor(pos / length), torch.tensor((pos+mask_length) / length)
        # left and right corners of boxes
        x1, x2 = bboxes[:,0] - bboxes[:,1]/2, bboxes[:,0] + bboxes[:,1]/2
        # part of box that was erased, not exactly IoU
        iou =  (x2.fmin(end) - x1.fmax(begin)).clip_(min=0) / (x2 - x1)
        # keep that labels 
        mask = iou < 0.4
        bboxes = bboxes[mask]
        labels = labels[mask]
        return signals, bboxes, labels

class ChannelMasking(Module):
    
    def __init__(self, p = 0.1):
        """
            Sets to zero random channels of signal with probability `p`.
        """
        super().__init__()
        self.p = p
    
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bernoulli_tensor = torch.bernoulli(torch.full(size=(signals.size(0),), fill_value=self.p)).bool()
        signals[bernoulli_tensor, :] = 0
        return signals, bboxes, labels
    
class ChannelScaling(Module):
    
    def __init__(self, sigma=0.4):
        """
            Applies random scale to each channel.

            Scale is generated from log-normal distribution:
            
            scale ~ Lognormal(0, `sigma`^2).
            
            sigma = 0.4 gives values in interval [0.3, 3.32] with probability 0.997.
        """
        super().__init__()
        self.sigma = sigma
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = np.random.lognormal(0, self.sigma, size=signals.size(0))
        scale = torch.tensor(scale).view(signals.size(0), -1)
        signals *= scale
        return signals, bboxes, labels
        
class MarkScaling(Module):
    
    def __init__(self, max_length, sigma=0.6):
        """
            Applies scaling to boxes whose size does not exceed `max_size`.
            
            scale ~ Lognormal(0, `sigma`^2).
            
            TODO: annotate
            
        """
        super().__init__()
        self.max_length = max_length
        self.sigma = sigma
    
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from scipy.signal.windows import hann
        
        signal_length = signals.shape[1]
        for box in bboxes:
            # compute size of box
            begin, end = box[0] - box[1]/2, box[0] + box[1]/2
            begin, end = int(begin * signal_length), int(end * signal_length)
            begin = max(begin,0)
            end = min(end, signal_length)
            box_length = end - begin
            if box_length > self.max_length:
                continue   # skip big boxes
            # window for continuous scaling 
            window = torch.tensor(hann(box_length), dtype=signals.dtype)
            # generate gandom scale
            scale = np.random.lognormal(0, self.sigma) - 1
            # scaling on box
            signals[:, begin:end] += scale * window * signals[:, begin:end]
        return signals, bboxes, labels
    

class PhaseSpectrumNoise(Module):
    
    def __init__(self, sigma:float = 0.5):
        """
            Adds noise in phase spectrum.
            
            Noise is from Normal distribution N(0,sigma^2).
        """
        super().__init__()
        self.sigma = sigma
        
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate the spectrum
        spectrum = torch.fft.rfft(signals, dim=-1)
        # extract magnitude and phase
        magnitude = torch.abs(spectrum)
        angles = torch.angle(spectrum)
        # add noise to phases
        noise = torch.randn_like(angles) * np.random.uniform(0, self.sigma)
        angles += noise
        # recover the spectrum
        complex_phase = torch.cos(angles) + 1j * torch.sin(angles)
        spectrum = magnitude * complex_phase
        # recover the signal
        signals = torch.fft.irfft(spectrum, dim=-1)
        return signals, bboxes, labels
        
class RandomApply(Module):
    def __init__(self, augmentation: Module, p:float=0.5):
            """
                Applies augmentation with probability.
            """
            super().__init__()
            self.augmentation = augmentation
            self.p = p 
            
    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if np.random.uniform(0, 1) < self.p:
            signals, bboxes, labels = self.augmentation(signals, bboxes, labels)
        return signals, bboxes, labels


class SineNoise(Module):

    def __init__(self, low: float = 0.002, high: float = 0.03, amplitude: Tuple[float, float] = (0.5, 3.0),
                 sigma: Tuple[float, float] = (0.05, 0.2), freq_jitter: float = 0.1, knot_step: int = 5000):
        """
            Adds a low-frequency sinusoidal interference A(c) * sin(phi(t)) to neighbouring channels.

            Frequency: omega_0 ~ U[`low`, `high`] rad/sample. Frequency and phase drift slowly along the
            signal: omega(t) = omega_0 * (1 + delta(t)), phi(t) = phi_0 + cumsum(omega(t)), phi_0 ~ U[0, 2*pi);
            delta(t) is a piecewise-linear random process with knots every `knot_step` samples and values
            U[-`freq_jitter`, `freq_jitter`], so the instantaneous frequency stays within
            omega_0 * (1 ± freq_jitter).

            Channel profile: channels are placed at c_k = (k - (C - 1) / 2) / C (for C = 15:
            -7/15 ... 7/15); m ~ U[-0.5, 0.5], s ~ U[`sigma`]; A(c_k) = a * exp(-(c_k - m)^2 / (2 s^2)) —
            the normal density normalised to its peak, a ~ U[`amplitude`] (units of the input signal;
            in the eddy pipeline — standardised signal, step 5 in TASKS.md). A narrow s puts the
            interference on a few neighbouring channels.
        """
        super().__init__()
        if not 0 < low <= high:
            raise ValueError(f"SineNoise: expected 0 < low <= high, got low={low}, high={high}")
        self.low, self.high = low, high
        self.amplitude = tuple(amplitude)
        self.sigma = tuple(sigma)
        self.freq_jitter = freq_jitter
        self.knot_step = knot_step

    def sample(self, n_channels: int, length: int):
        """Random parameters of one interference: (profile (C,), omega(t) (L,), phi(t) (L,), omega_0)."""
        omega0 = np.random.uniform(self.low, self.high)
        n_knots = max(2, int(np.ceil(length / self.knot_step)) + 1)
        knots = np.random.uniform(-self.freq_jitter, self.freq_jitter, size=n_knots)
        delta = np.interp(np.arange(length), np.linspace(0, length - 1, n_knots), knots)
        omega = omega0 * (1 + delta)
        phi = np.random.uniform(0, 2 * np.pi) + np.cumsum(omega)
        coords = (np.arange(n_channels) - (n_channels - 1) / 2) / n_channels
        m = np.random.uniform(-0.5, 0.5)
        s = np.random.uniform(*self.sigma)
        profile = np.random.uniform(*self.amplitude) * np.exp(-(coords - m) ** 2 / (2 * s ** 2))
        return profile, omega, phi, omega0

    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        profile, _, phi, _ = self.sample(signals.size(0), signals.size(1))
        noise = torch.from_numpy(profile[:, None] * np.sin(phi)[None, :]).to(signals.dtype)
        return signals + noise, bboxes, labels