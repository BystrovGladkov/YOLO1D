"""Вихретоковые дефектограммы: предобработка (подавление помех, пороговые уровни, замена битых каналов)."""
from .preprocess import PreprocessResult, preprocess, prepare_signal, substitute_channels, fftine_fast, convert_freq

__all__ = ["PreprocessResult", "preprocess", "prepare_signal", "substitute_channels", "fftine_fast", "convert_freq"]
