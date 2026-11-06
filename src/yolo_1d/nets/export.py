import torch
from torch.nn import Module

from yolo_1d.utils.constants import *

class ExportWrapper(Module):
    """
        Wrapper over the model to use in exported version.

        resize=True (вихретоковые данные): signal is interpolated to `input_length` as preprocessing
        step and predicted boxes are scaled back to original coordinates as postprocessing step.

        resize=False (ЭКГ): signal is zero-padded on the right to `input_length`; predicted boxes are
        already in samples of the original signal.

        Модель с n_leads > 1 (режим MC): выход (B·n_leads, C, A) переупорядочивается в (B, n_leads, C, A).
    """

    def __init__(self, model, input_length: int = INPUT_LENGTH, resize: bool = True):
        super().__init__()
        self.model = model
        self.input_length = input_length
        self.resize = resize
        model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.resize:
            # pad to network input length without changing time scale
            x = torch.nn.functional.pad(x, (0, self.input_length - x.size(dim=-1)))
            return self._per_lead(self.model(x), x.size(0))
        input_length = x.size(dim=-1)
        # perform interpolation to desired length
        x = torch.nn.functional.interpolate(x, self.input_length, mode='linear')
        # get predictions
        output = self.model(x)
        # scale the box size
        output[:, :2, :] = output[:, :2, :] / self.input_length * input_length
        return self._per_lead(output, x.size(0))

    def _per_lead(self, output: torch.Tensor, batch: int) -> torch.Tensor:
        n_leads = getattr(self.model, "n_leads", 1)
        return output if n_leads == 1 else output.view(batch, n_leads, *output.shape[1:])
