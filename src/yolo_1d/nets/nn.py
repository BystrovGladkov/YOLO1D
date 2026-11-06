import math

import torch
import torch.nn as nn

from yolo_1d.nets.ops import make_anchors
from yolo_1d.utils.constants import *

class ConvBlock(nn.Module): 
    """
        Базовый слой свёртки с BatchNorm и Функцией активации
    """
    
    def __init__(self, channels_in, channels_out, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        # Одномерная свёртка. 
        self.convolution = nn.Conv1d(channels_in, channels_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=False, groups=g)
        # Слой нормализации по батчу.
        self.batch_normalization = nn.BatchNorm1d(channels_out, eps=0.001, momentum=0.03)
        # Гладкое прилижение ReLU
        self.activation = activation# nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.activation(self.batch_normalization(self.convolution(x)))


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, factor=0.5):
        super().__init__()
        self.convolution1 = ConvBlock(channels, int(channels * factor), torch.nn.SiLU(), k=3, p=1)
        self.convolution2 = ConvBlock(int(channels * factor), channels, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.convolution2(self.convolution1(x))


class C3K(torch.nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.convolution1 = ConvBlock(channels_in, channels_out // 2, torch.nn.SiLU())
        self.convolution2 = ConvBlock(channels_in, channels_out // 2, torch.nn.SiLU())
        self.convolution3 = ConvBlock(2 * (channels_out // 2), channels_out, torch.nn.SiLU())
        self.residuals = torch.nn.Sequential(ResidualBlock(channels_out // 2, factor=1.0),
                                         ResidualBlock(channels_out // 2, factor=1.0))

    def forward(self, x):
        y = self.residuals(self.convolution1(x))
        return self.convolution3(torch.cat((y, self.convolution2(x)), dim=1))


class C3K2(torch.nn.Module):
    def __init__(self, channels_in, channels_out, depth, use_csp, factor):
        super().__init__()
        self.convolution1 = ConvBlock(channels_in, 2 * (channels_out // factor), torch.nn.SiLU())
        self.convolution2 = ConvBlock((2 + depth) * (channels_out // factor), channels_out, torch.nn.SiLU())

        if use_csp:
            self.res_m = torch.nn.ModuleList(C3K(channels_out // factor, channels_out // factor) for _ in range(depth))
        else:
            self.res_m = torch.nn.ModuleList(ResidualBlock(channels_out // factor) for _ in range(depth))
            

    def forward(self, x):
        y = list(self.convolution1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.convolution2(torch.cat(y, dim=1))


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, channels, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = channels // self.num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.convolution1 = ConvBlock(channels, channels, torch.nn.Identity(), k=3, p=1, g=channels)
        self.convolution2 = ConvBlock(channels, channels, torch.nn.Identity())
        
        self.qkv = ConvBlock(channels, channels + self.dim_key * self.num_head * 2, torch.nn.Identity())

    def forward(self, x):
        b, c, l = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, l)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, l) + self.convolution1(v.reshape(b, c, l))
        return self.convolution2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, channels, num_head):
        super().__init__()
        self.attention = MultiHeadAttention(channels, num_head)
        self.feed_forward = torch.nn.Sequential(ConvBlock(channels, channels * 2, torch.nn.SiLU()),
                                                ConvBlock(channels * 2, channels, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class C2PSA(torch.nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.convolution1 = ConvBlock(channels, 2 * (channels // 2), torch.nn.SiLU())
        self.convolution2 = ConvBlock(2 * (channels // 2), channels, torch.nn.SiLU())
        self.attention = torch.nn.Sequential(*(PSABlock(channels // 2, channels // 128) for _ in range(depth)))

    def forward(self, x):
        x, y = self.convolution1(x).chunk(2, 1)
        return self.convolution2(torch.cat(tensors=(x, self.attention(y)), dim=1))

class SPPF(torch.nn.Module):
    def __init__(self, channels_in, channels_out, k=5):
        super().__init__()
        self.convolution1 = ConvBlock(channels_in, channels_in // 2, torch.nn.SiLU())
        self.convolution2 = ConvBlock(channels_in * 2, channels_out, torch.nn.SiLU())
        self.max_pooling = torch.nn.MaxPool1d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.convolution1(x)
        x1 = self.max_pooling(x)
        x2 = self.max_pooling(x1)
        x3 = self.max_pooling(x2)
        return self.convolution2(torch.cat(tensors=[x, x1, x2, x3], dim=1))


# Шаблоны шагов 8 «даунсэмплирующих» свёрток бэкбона (docs/research/R4_architecture.md §4).
# Страйды голов = (произведение первых 5, первых 7, всех 8 шагов).
STRIDE_TEMPLATES = {
    "original": [2, 2, 2, 2, 2, 2, 2, 2],  # {32, 128, 256} — вихретоковые данные (вход 49152)
    "A": [2, 2, 1, 1, 1, 2, 2, 2],         # {4, 16, 32}  — ЭКГ, рекомендован (R4 §5.1)
    "B": [2, 2, 2, 1, 1, 1, 2, 2],         # {8, 16, 32}  — ЭКГ, запасной
    "C": [2, 1, 1, 1, 2, 1, 2, 2],         # {4, 8, 16}   — ЭКГ, альтернатива (R4 §5.2)
}


def resolve_strides(strides=None):
    """Имя шаблона или список из 8 шагов (1 или 2) → список шагов. None → "original"."""
    if strides is None:
        strides = "original"
    if isinstance(strides, str):
        if strides not in STRIDE_TEMPLATES:
            raise ValueError(f"неизвестный шаблон страйдов {strides!r}; доступны {list(STRIDE_TEMPLATES)}")
        return list(STRIDE_TEMPLATES[strides])
    strides = [int(s) for s in strides]
    if len(strides) != 8 or any(s not in (1, 2) for s in strides):
        raise ValueError(f"нужно 8 шагов из {{1, 2}}, получено {strides}")
    return strides


def head_strides(strides=None):
    """Шаги 8 свёрток → страйды трёх уровней детекции (p1, p2, p3)."""
    s = resolve_strides(strides)
    return math.prod(s[:5]), math.prod(s[:7]), math.prod(s)


class Backbone(torch.nn.Module):
    def __init__(self, input_channels, width, depth, use_csp, strides=None):
        """
            strides
                Шаги 8 свёрток понижения разрешения (имя шаблона из STRIDE_TEMPLATES или список).
                По умолчанию — исходная конфигурация: все шаги 2, страйды голов {32, 128, 256}.
                Шаг 1 меняет только разрешение: параметры и имена слоёв те же.
        """
        super().__init__()
        s = resolve_strides(strides)
        self.step1 = []
        self.step2 = []
        self.step3 = []

        # p1/2
        # 50000
        self.step1.append(ConvBlock(input_channels, width[0], torch.nn.SiLU(), 3, s[0], 1))      # 24576, 3
        self.step1.append(ConvBlock(width[0], width[1], torch.nn.SiLU(), 3, s[1], 1))      # 12288, 7
        self.step1.append(ConvBlock(width[1], width[2], torch.nn.SiLU(), 3, s[2], 1))     # 6144, 19
        self.step1.append(C3K2( width[2], width[2], depth[0], use_csp[0], factor=2))                               # 6144, ? 40
        self.step1.append(ConvBlock(width[2], width[3], torch.nn.SiLU(), 3, s[3], 1))    # 3072  ? 80
        self.step1.append(ConvBlock(width[3], width[4], torch.nn.SiLU(), 3, s[4], 1))    # 1536  ? 200
        self.step1.append(C3K2( width[4], width[4], depth[1], use_csp[0], factor=2))                               # 1536  ? 300
        self.step1 = torch.nn.Sequential(*self.step1)

        # p2/4
        self.step2.append(ConvBlock(width[4], width[5], torch.nn.SiLU(), 3, s[5], 1))    # 768, 700
        self.step2.append(ConvBlock(width[5], width[6], torch.nn.SiLU(), 3, s[6], 1))    # 384, 1500
        self.step2.append(C3K2( width[6], width[6], depth[2], use_csp[0], factor=2))                               # 384, 3000
        self.step2 = torch.nn.Sequential(*self.step2)
        
        # p3/8
        self.step3.append(ConvBlock(width[6], width[7], torch.nn.SiLU(), 3, s[7], 1))    # 195, 8000
        self.step3.append(C3K2( width[7], width[7], depth[3], use_csp[0], factor=2))                               # 195, 12000
        self.step3.append(SPPF(width[7], width[7]))
        self.step3.append(C2PSA(width[7], depth[4]))
        self.step3 = torch.nn.Sequential(*self.step3)

    def forward(self, x):
        x1 = self.step1(x)
        x2 = self.step2(x1)
        x3 = self.step3(x2)
        return x1, x2, x3


def _downsample(ch, ratio):
    """Понижение разрешения в `ratio` раз свёртками с шагом 2 (ratio ∈ {1, 2, 4, 8}).

    Для ratio = 2 — одиночный Conv, для ratio ≥ 4 — Sequential: при отношениях исходной сети
    (4 и 2) имена параметров совпадают с прежними (h3.0, h3.1, h5) и старые чекпоинты загружаются.
    """
    n = {1: 0, 2: 1, 4: 2, 8: 3}.get(ratio)
    if n is None:
        raise ValueError(f"отношение страйдов соседних уровней должно быть 1, 2, 4 или 8, получено {ratio}")
    if n == 0:
        return torch.nn.Identity()
    convs = [ConvBlock(ch, ch, torch.nn.SiLU(), 3, 2, 1) for _ in range(n)]
    return convs[0] if n == 1 else nn.Sequential(*convs)


class Neck(torch.nn.Module):
    def __init__(self, width, depth, csp, ratio12=4, ratio23=2):
        """
            ratio12, ratio23
                Отношения страйдов уровней p2/p1 и p3/p2. По умолчанию — исходная сеть (4 и 2).
        """
        super().__init__()
        self.upsample1 = torch.nn.Upsample(scale_factor=ratio23)
        self.upsample2 = torch.nn.Upsample(scale_factor=ratio12)
        self.c3k2_1 = C3K2(width[6] + width[7], width[7], depth[4], csp[1], factor=2)
        self.c3k2_2 = C3K2(width[4] + width[7], width[7], depth[5], csp[1], factor=2)
        self.convolution1 = _downsample(width[7], ratio12)
        self.c3k2_3 = C3K2(width[7] * 2, width[7], depth[6], csp[1], factor=2)
        self.convolution2 = _downsample(width[7], ratio23)
        self.c3k2_4 = C3K2(width[7] * 2, width[7], depth[7], csp[2], factor=2)

    def forward(self, x):
        x1, x2, x3 = x

        x2 = self.c3k2_1(torch.cat(tensors=[self.upsample1(x3), x2], dim=1))

        x1 = self.c3k2_2(torch.cat(tensors=[self.upsample2(x2), x1], dim=1))

        x2 = self.c3k2_3(torch.cat(tensors=[self.convolution1(x1), x2], dim=1))

        x3 = self.c3k2_4(torch.cat(tensors=[self.convolution2(x2), x3], dim=1))
        return x1, x2, x3


class DFL(torch.nn.Module):
    def __init__(self, reg_max=16):
        # ch * stride_of_head  = максимальный размер предсказываемого объекта
        super().__init__()
        self.reg_max = reg_max
        self.frozen_conv = torch.nn.Conv2d(self.reg_max, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(reg_max, dtype=torch.float).view(1, self.reg_max, 1, 1)
        self.frozen_conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 2, self.reg_max, a).transpose(2, 1) # Теперь предсказывается только начало и конец
        return self.frozen_conv(x.softmax(1)).view(b, 2, a)


class Head(torch.nn.Module):
    input_length = INPUT_LENGTH  # длина входа сети; задаётся моделью YOLO

    def __init__(self, num_classes, input_channels=(), n_leads: int = 1, width_mult: float = 1.0):
        """
            n_leads
                Число наборов предсказаний (режим MC, M7): выходные 1×1-свёртки box и cls расширяются в
                n_leads раз, промежуточные слои голов общие. Выход переупорядочивается в «батч отведений»
                (B·n_leads, no, A): строка b·n_leads + k — предсказания отведения k примера b. При n_leads = 1
                веса и выход совпадают с исходной головой.
            width_mult
                Множитель ширины промежуточных слоёв веток box и cls (1 — как раньше).
        """
        super().__init__()
        self.reg_max = 16
        self.num_classes = num_classes
        self.n_leads = n_leads
        self.n_heads = len(input_channels)  # number of input channels for each Head
        self.num_outputs = num_classes + self.reg_max * 2  # number of outputs per anchor (per lead)
        self.stride = torch.zeros(self.n_heads)  # strides computed during build
        self.validate = False # flag for validation - model returns raw and processed predictions

        box_channels = int(round(max(64, input_channels[0] // 4) * width_mult))
        cls_channels = int(round(max(80, input_channels[0], self.num_classes) * width_mult))

        self.dfl = DFL(self.reg_max)
        self.mlp_box = torch.nn.ModuleList(torch.nn.Sequential(ConvBlock(x, box_channels,torch.nn.SiLU(), k=3, p=1),
                                                           ConvBlock(box_channels, box_channels,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv1d(box_channels, out_channels=2 * self.reg_max * n_leads,
                                                                           kernel_size=1)) for x in input_channels)
        self.mlp_cls = torch.nn.ModuleList(torch.nn.Sequential(ConvBlock(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                           ConvBlock(x, cls_channels, torch.nn.SiLU()),
                                                           ConvBlock(cls_channels, cls_channels, torch.nn.SiLU(), k=3, p=1, g=cls_channels),
                                                           ConvBlock(cls_channels, cls_channels, torch.nn.SiLU()),
                                                           torch.nn.Conv1d(cls_channels, out_channels=self.num_classes * n_leads,
                                                                           kernel_size=1)) for x in input_channels)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.mlp_box, self.mlp_cls)):
            b, c = box(x[i]), cls(x[i])
            n, _, a = b.shape
            # (B, L·2ch, A), (B, L·nc, A) -> (B·L, 2ch + nc, A); при L = 1 — то же, что cat по каналам
            x[i] = torch.cat(tensors=(b.view(n, self.n_leads, 2 * self.reg_max, a),
                                      c.view(n, self.n_leads, self.num_classes, a)), dim=2).view(n * self.n_leads, self.num_outputs, a)
        if self.training:
            return x

        anchors, strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        y = torch.cat([i.view(x[0].shape[0], self.num_outputs, -1) for i in x], dim=2)
        box, cls = y.split(split_size=(2 * self.reg_max, self.num_classes), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        y = torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)
        
        return (x, y) if self.validate else y

    def initialize_biases(self, mode: str = "legacy", expected_objects: float = 24.0):
        """
            Initialize biases. WARNING: requires stride availability.

            mode
                "legacy" — исходная формула из 2D YOLO: log(5 / nc / (L / s)²) (квадрат числа ячеек);
                сохранена для воспроизводимости вихретоковых экспериментов.
                "1d" — априорная вероятность объекта на якорь в 1D:
                log(expected_objects / nc / Σ_l (L / s_l)), где Σ_l (L / s_l) — число якорей.
            expected_objects
                Ожидаемое число объектов на вход (для "1d"); LUDB — ≈ 24 волны на отведение.
        """
        if mode not in ("legacy", "1d"):
            raise ValueError(f"mode должен быть 'legacy' или '1d', получено {mode!r}")
        num_anchors = sum(float(self.input_length / s) for s in self.stride)
        for box, cls, s in zip(self.mlp_box, self.mlp_cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            if mode == "legacy":
                # cls (.01 objects, 80 classes, 640 image)
                cls[-1].bias.data[:] = math.log(5 / self.num_classes / (self.input_length / s) ** 2)
            else:
                # одинаково для каждого из n_leads наборов (априорная вероятность на якорь отведения)
                cls[-1].bias.data[:] = math.log(expected_objects / self.num_classes / num_anchors)


class YOLO1D(torch.nn.Module):
    def __init__(self, width=None, depth=None, use_csp=None, num_classes=3, input_channels=15,
                 input_length: int = INPUT_LENGTH, strides=None, bias_init: str = "legacy",
                 expected_objects: float = 24.0, n_leads: int = 1, head_width_mult: float = 1.0):
        """
            input_channels
                Число каналов входного сигнала (30 — вихретоковые данные, 1 — ЭКГ по отведениям).
            input_length
                Длина входа сети, по которой вычисляются страйды голов (49152 — вихретоковые данные,
                5120 — ЭКГ). Должна делиться на максимальный страйд сети.
            strides
                Шаги свёрток бэкбона: имя шаблона из STRIDE_TEMPLATES ("original", "A", "B", "C")
                или список из 8 шагов. По умолчанию — исходная сеть {32, 128, 256}.
            bias_init, expected_objects
                Инициализация смещений классификации (см. Head.initialize_biases).
            n_leads, head_width_mult
                Режим MC (M7): число наборов предсказаний головы (по отведению) и ширина её веток (см. Head).
                Выход — «батч отведений» (B·n_leads, ...).
        """
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.n_leads = n_leads
        s1, s2, s3 = head_strides(strides)
        if input_length % s3:
            raise ValueError(f"длина входа {input_length} должна делиться на максимальный страйд {s3}")
        self.backbone = Backbone(input_channels, width, depth, use_csp, strides)
        self.neck = Neck(width, depth, use_csp, ratio12=s2 // s1, ratio23=s3 // s2)
        
        # calculate strides of each head
        signal_example = torch.zeros(1, input_channels, input_length)
        self.head = Head(num_classes, (width[7], width[7], width[7]), n_leads=n_leads, width_mult=head_width_mult)
        self.head.input_length = input_length
        self.head.stride = torch.tensor([input_length / x.shape[-1] for x in self.forward(signal_example)])
        self.stride = self.head.stride
        self.head.initialize_biases(bias_init, expected_objects)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(list(x))

    def set_validate(self, value: bool):
        """Sets model state to get raw (for loss computation) and decoded boxes (for metrics evaluation)."""
        self.head.validate = value


def yolo1d_n(num_classes: int, input_channels: int = 30, input_length: int = INPUT_LENGTH, **arch):
    csp = [False, False, True]
    depth = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    width = [32, 32, 32, 64, 128, 128, 128, 128]
    return YOLO1D(width, depth, csp, num_classes, input_channels, input_length, **arch)


def yolo1d_s(num_classes: int, input_channels: int = 30, input_length: int = INPUT_LENGTH, **arch):
    csp = [True, True, True]
    depth = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    width = [32, 32, 64, 96, 128, 128, 192, 192]
    return YOLO1D(width, depth, csp, num_classes, input_channels, input_length, **arch)


def yolo1d_m(num_classes: int, input_channels: int = 30, input_length: int = INPUT_LENGTH, **arch):
    csp = [True, True, True]
    depth = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    width = [32, 32, 64, 96, 128, 192, 192, 256]
    return YOLO1D(width, depth, csp, num_classes, input_channels, input_length, **arch)


def yolo1d_l(num_classes: int, input_channels: int = 30, input_length: int = INPUT_LENGTH, **arch):
    csp = [True, True, True]
    depth = [1, 1, 1, 1, 1, 2, 2, 2, 3]
    width = [32, 64, 64, 96, 128, 192, 256, 512]
    return YOLO1D(width, depth, csp, num_classes, input_channels, input_length, **arch)

def yolo1d_xl(num_classes: int, input_channels: int = 30, input_length: int = INPUT_LENGTH, **arch):
    csp = [True, True, True]
    depth = [1, 3, 3, 3, 3, 3, 4, 4, 5]
    width = [32, 64, 64, 96, 128, 192, 256, 512]
    return YOLO1D(width, depth, csp, num_classes, input_channels, input_length, **arch)

def get_model(size: str, num_classes: int, input_channels: int = 30, input_length: int = INPUT_LENGTH, **arch):
    """
        Создаёт модель размера size ("nano", "small", "medium", "large", "xlarge").
        По умолчанию — конфигурация для вихретоковых данных (30 каналов, вход 49152).
        arch — strides, bias_init, expected_objects, n_leads, head_width_mult (см. YOLO1D).
    """
    if size == "nano":
        return yolo1d_n(num_classes, input_channels, input_length, **arch)
    if size == "small":
        return yolo1d_s(num_classes, input_channels, input_length, **arch)
    if size == "medium":
        return yolo1d_m(num_classes, input_channels, input_length, **arch)
    if size == "large":
        return yolo1d_l(num_classes, input_channels, input_length, **arch)
    if size == "xlarge":
        return yolo1d_xl(num_classes, input_channels, input_length, **arch)