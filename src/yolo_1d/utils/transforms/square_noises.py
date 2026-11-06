import numpy as np
from scipy import signal
from scipy.stats import expon
from numpy.fft import rfft, irfft, rfftfreq
from scipy.stats import median_abs_deviation as mad
from numba import njit
from sklearn.decomposition import FastICA
from typing import Tuple
import torch
from torch.nn import Module
import math

def havierian(x, fun_args=None):
    k1 = 36.0 / (8.0 * np.sqrt(3.0) - 9.0)
    k2 = 24.0 / (16.0 * np.sqrt(3.0) - 27.0)
    c = np.sqrt(0.5)
    
    exp_x = np.exp(-x**2 / 2.0)
    G1_val = x * exp_x
    g1_val = (1.0 - x**2) * exp_x
    g1_prime = x * (x**2 - 3.0) * exp_x
    
    E_G1 = np.mean(G1_val, axis=1) # Вектор (n_components,)
    
    # --- Вычисляем четную функцию G2 (хвосты) ---
    # G2(y) = exp(-y^2 / 2)
    G2_val = exp_x
    g2_val = -x * exp_x
    g2_prime = (x**2 - 1.0) * exp_x
    
    E_G2 = np.mean(G2_val, axis=1) # Вектор (n_components,)
    
    # --- Динамические веса градиентов для КАЖДОГО компонента ---
    beta1 = k1 * E_G1
    beta2 = k2 * (E_G2 - c)
    
    # Broadcasting весов к форме (n_components, n_samples)
    beta1_mat = beta1[:, np.newaxis]
    beta2_mat = beta2[:, np.newaxis]
    
    g_comb = beta1_mat * g1_val + beta2_mat * g2_val
    g_prime_comb = beta1_mat * g1_prime + beta2_mat * g2_prime
    return g_comb, g_prime_comb.mean(axis=-1)

def new_smooth_noise(N = 50000, dt = 1):
    # Распределения амплитуд
    mu = np.array([9.5, -9.5])
    Rho = np.array([[5, 0], [0, 5]])

    m_vec = np.random.multivariate_normal(mu, Rho)

    # Телеграфный процесс
    # lambda_avg = 0.0475
    # lambda_var = 0.001
    # theta_param = lambda_avg / lambda_var 
    # gamma_param = lambda_avg * theta_param
    # lambda_vec = gamma.rvs(gamma_param, scale = 1.0 / theta_param, size = 2)
    lambda_vec = [0.046, 0.046]

    # Марковская макро-цепь (шумовые сегменты)
    # Интенсивности перехода из чистого состояния
    lambda_01 = 0.00008     # Возникновение короткого шума
    lambda_02 = 0.00006      # Возникновение длинного шума (в 4 раза реже короткого)

    # Полная интенсивность выхода из 0
    Lambda_0 = lambda_01 + lambda_02
    prob_short = lambda_01 / Lambda_0 # Вероятность перехода в нулевое состояние

    lambda_short = 0.00474
    lambda_long = 2.857e-5

    # Сглаживание
    # RC-фильтр (экспонента в конце фрагмента)
    lambda_rc = 0.5
    tau_rc = expon.rvs(scale = 1.0 / lambda_rc)

    #print(f"m_vec = {m_vec}, lambda_vec = {lambda_vec}, tau_rc = {tau_rc}")

    # Фильтр нижних частот
    omega_c = 0.898
    # Перевод в частоту Найквиста. Из отрезка [0; pi] в [0; 1]
    nyq = omega_c / np.pi

    # Фильтр Баттерворта 2-го порядка
    b2, a2 = signal.butter(2, nyq, btype='low')

    # ----- НАЧАЛО СИМУЛЯЦИИ -----
    T = N / dt # Время симуляции

    # Симуляция Макро-цепи (генерация маски z)
    t = 0.0
    macro_state = 0  # 0: чисто, 1: короткий шум, 2: длинный шум
    macro_events = []  # Хранит (t_start, t_end, state)

    while t < T:
        if macro_state == 0:
            dT = expon.rvs(scale = 1.0 / Lambda_0)
            t_next = t + dT
            macro_events.append((t, min(t_next, T), 0))
            macro_state = 1 if np.random.rand() < prob_short else 2
        elif macro_state == 1:
            dT = expon.rvs(scale = 1.0 / lambda_short)
            t_next = t + dT
            macro_events.append((t, min(t_next, T), 1))
            macro_state = 0
        elif macro_state == 2:
            dT = expon.rvs(scale = 1.0 / lambda_long)
            t_next = t + dT
            macro_events.append((t, min(t_next, T), 2))
            macro_state = 0
        t = t_next

    # Симуляция телеграфного процесса
    micro_events = []  # Хранит (t_start, t_end, amplitude_value)

    for t_start, t_end, state in macro_events:
        if state == 0:
            continue 

        t_sub = t_start
        # Начальное состояние
        micro_state = np.random.choice([0, 1])

        while t_sub < t_end:
            rate = lambda_vec[micro_state]
            dt_sub = expon.rvs(scale = 1.0 / rate)
            t_sub_next = t_sub + dt_sub

            val = m_vec[micro_state]
            micro_events.append((t_sub, min(t_sub_next, t_end), val))

            t_sub = t_sub_next
            micro_state = 1 - micro_state  # Переключение 0 -> 1 или 1 -> 0

    # Формирование итоговых массивов с данными и шумовыми сегментами
    z = np.zeros(N, dtype=float)
    x = np.zeros(N, dtype=float)

    # Формируем бинарную маску фрагментов шума
    for t_start, t_end, state in macro_events:
        if state in (1, 2):
            idx_start = max(0, min(N, int(t_start * dt)))
            idx_end = max(0, min(N, int(t_end * dt)))
            z[idx_start:idx_end] = 1.0

    # Заполняем амплитуды несглаженного телеграфного процесса
    for t_start, t_end, val in micro_events:
        idx_start = max(0, min(N, int(t_start * dt)))
        idx_end = max(0, min(N, int(t_end * dt)))
        x[idx_start:idx_end] = val

    x_sym = np.zeros_like(x)

    harmonics = rfft(x)
    freqs = rfftfreq(N)
    porog = omega_c / 2 / np.pi
    harmonics[freqs > porog] = 0
    x_smooth = irfft(harmonics, n=N)

    return x, z, x_smooth


def ICA_filtration(data, median = False, EMA = False, method = 'cube', tol = 0.01, max_iter = 100,
                   w_1_length = 15, w_2_length = 80, threshold1 = 0.03227, threshold2 = 0.04741,
                   close_w_length = 120, open_w_length = 70, percent_error = 0.14008,
                   threshold1_cor = 0.066134, threshold2_cor = 0.09577, median_w_size = 13,
                   sigma_med_coef = 2.241, robust_sigma = 1.6098):
    '''
    Фильтрация в базисе ICA
    
    :param data: Дефектограмма
    :param median: Требуется ли медианная фильтрация?
    :param tol, max_iter: критерий остановки FastICA — как в продакшен-реализации на C++ (0.01 и 100)
    '''
    if method == "hyvarinen":
        ica = FastICA(n_components=5, whiten="unit-variance", whiten_solver="svd", tol=tol, max_iter=max_iter, fun=havierian, random_state=55)
    else:
        ica = FastICA(n_components=5, whiten="unit-variance", whiten_solver="svd", tol=tol, max_iter=max_iter, fun=method, random_state=55)
    
    signals = ica.fit_transform(data.T).T
    residual = data - ica.inverse_transform(signals.T).T

    small_window = np.zeros_like(signals, dtype=np.float32)
    big_window = np.zeros_like(signals, dtype=np.float32)
    for i in range(0, signals.shape[0]):
            small_window[i,:] = SW_criteria_exp(signals[i], w_1_length)
            big_window[i,:] = SW_criteria_exp(signals[i], w_2_length)
            
            # small_window[i,:] = SW_criteria_optimized(signals[i], 30, 1)
            # big_window[i,:] = SW_criteria_optimized(signals[i], 75, 1)
    
    signals_filtered = signals
    if(median):
        median_mask = np.zeros_like(signals, dtype=bool)
        for i in range(0, signals.shape[0]):
            # my_median_filter(signals_filtered[i, :],  median_mask[i, :],
                            # window_size=25, sigma_coef=2)
            exponential_median_filter(signals_filtered[i, :],  median_mask[i, :],
                                window_size=median_w_size, sigma_coef=sigma_med_coef, robust_sigma=robust_sigma)
        rez = bad_data_morphology(small_window, big_window, threshold1, threshold2, close_w_length, open_w_length, median_mask)
        need_bigger_thresh = np.sum(rez, axis=1) > data.shape[1]*percent_error
        if need_bigger_thresh.any():
            rez[need_bigger_thresh] = bad_data_morphology(small_window[need_bigger_thresh], big_window[need_bigger_thresh], threshold1_cor,
                                                          threshold2_cor, close_w_length, open_w_length, median_mask[need_bigger_thresh])
    else:
        rez = bad_data_morphology(small_window, big_window, threshold1, threshold2, close_w_length, open_w_length)
        need_bigger_thresh = np.sum(rez, axis=1) > data.shape[1]*percent_error
        if need_bigger_thresh.any():
            rez[need_bigger_thresh] = bad_data_morphology(small_window[need_bigger_thresh], big_window[need_bigger_thresh], threshold1_cor,
                                                          threshold2_cor, close_w_length, open_w_length)
    
    
    signals_filtered = signals_filtered * (1 - rez)

    result = residual + ica.inverse_transform(signals_filtered.T).T
    # signals_filtered = signals_filtered * (1 - impulse)[:,None]
    return result


@njit
def SW_criteria_exp(data: np.ndarray, window=50):
    """
    Вычисляет SW-статистику сигнала с оптимизацией.
    """
    arr = np.ones_like(data, dtype=np.float32)
    alpha = 2 / (window * 2 + 1)
    
    p = 0.5 # вероятность верхнего импульса
    err = 0.0   # скользящая квадратичная ошибка
    EX_exp = float(data[0]) # Скользящее среднее
    EX1_exp = float(data[0]) # скользящее среднее EX1
    EX2_exp = float(data[0]) # скользящее среднее EX2
    # начальная инициализация среднего
    for i in range(0, window):
        EX_exp = EX_exp * (1 - alpha) + data[i] * alpha
    
    for i in range(0, window):
        EX_exp = EX_exp * (1 - alpha) + data[i + window] * alpha
        if data[i] < EX_exp:
            p = p * (1 - alpha) + alpha * 0
            alpha_1 = 2 / (window * 2 * (1-p))
            EX1_exp = EX1_exp * (1 - alpha_1) + data[i] * alpha_1
            err = err * (1 - alpha) + np.square(data[i] - EX1_exp) * alpha
        else:
            p = p * (1 - alpha) + alpha * 1
            alpha_1 = 2 / (window * 2 * p)
            EX2_exp = EX2_exp * (1 - alpha_1) + data[i] * alpha_1
            err = err * (1 - alpha) + np.square(data[i] - EX2_exp) * alpha
    
    for i in range(window, data.shape[0] - window):
        # Итеративное обновление среднего
        EX_exp = EX_exp * (1 - alpha) + data[i + window] * alpha
        
        if data[i] < EX_exp: # нижний импульс
            # обновление вероятности верхнего импульса
            p = p * (1 - alpha)
            # вычисление параметров для EMA
            alpha_1 = 2 / (window * 2 * (1-p))
            EX1_exp = EX1_exp * (1 - alpha_1) + data[i] * alpha_1
            # скользящая ошибка
            err = err * (1 - alpha) + np.square(data[i] - EX1_exp) * alpha
        else: # верхний импульс
            # обновление вероятности верхнего импульса
            p = p * (1 - alpha) + alpha
            # вычисление параметров для EMA
            alpha_1 = 2 / (window * 2 * p)
            EX2_exp = EX2_exp * (1 - alpha_1) + data[i] * alpha_1
            # скользящая ошибка
            err = err * (1 - alpha) + np.square(data[i] - EX2_exp) * alpha
        arr[i - window] = min(1, err / (np.square(EX1_exp - EX2_exp)+1e-7))
    return arr

def bad_data_morphology(small_window, big_window, small_window_threshold, big_window_threshold, close_window, open_window, median_mask = None):
    a = small_window < small_window_threshold
    b = big_window < big_window_threshold
    
    rez = a | b
    if(median_mask is not None):
        rez = rez | median_mask
    
    return close_opening(rez, close_window, open_window)

def close_opening(rez: np.array, close_window, open_window):
    """Морфологическое закрытие (окно close_window), затем открытие (окно open_window) бинарной маски по оси 1.

    Эквивалентно прежней реализации на `scipy.ndimage.convolve1d(..., mode="nearest")` с окном из единиц
    (побитно, tests/test_robust_segmenter.py): дилатация — «в окне есть единица», эрозия — «в окне все единицы».
    """
    rez = np.ascontiguousarray(rez, dtype=np.bool_)
    # close
    rez = _window_count_ge(rez, close_window, 1)                # dilation
    rez = _window_count_ge(rez, close_window, close_window)     # erosion
    # open
    rez = _window_count_ge(rez, open_window, open_window)       # erosion
    rez = _window_count_ge(rez, open_window, 1)                 # dilation
    return rez


@njit(cache=True)
def _window_count_ge(mask, window, threshold):
    """out[r, i] = (число единиц mask[r] в окне [i − (window − 1) // 2, i + window // 2]) >= threshold.

    Окно — как у `convolve1d` с ядром из `window` единиц; за краями сигнал продолжается крайним значением
    (mode="nearest"). Скользящая сумма — O(n) на строку.
    """
    rows, n = mask.shape
    out = np.empty((rows, n), dtype=np.bool_)
    left = (window - 1) // 2
    right = window // 2
    for r in range(rows):
        count = 0
        for k in range(-left, right + 1):
            count += mask[r, min(max(k, 0), n - 1)]
        out[r, 0] = count >= threshold
        for i in range(1, n):
            count += mask[r, min(i + right, n - 1)]
            count -= mask[r, max(i - 1 - left, 0)]
            out[r, i] = count >= threshold
    return out


def exponential_median_filter(signal: np.ndarray, mask: np.ndarray, window_size: int, sigma_coef = 2, robust_sigma = 2):
    exp_median_filter = OnlineRobustSegmenter(window_size=window_size, sigma_coef=sigma_coef, robust_sogma_coef=robust_sigma)
    return exp_median_filter.process_signal(mask, signal)


class OnlineRobustSegmenter:
    def __init__(self, window_size=20, sigma_coef=2.0, robust_sogma_coef=2.0):
        """
            Класс для подавления импульсных шумов. Вычисляет MAD с использованием приближения через экспоненциальное среднее.
        
        :param window_size: Размер окна, в котором вычисляется медиана
        :param sigma_coef: Параметр для определения отсчёта аномальным
        :param robust_sogma_coef: Параметр для вычисления робастной оценки
        """
        
        # Переводим размер окна в alpha для EMA
        # alpha = 2 / (N + 1) — стандартная формула
        self.window_size = window_size
        self.alpha = 2.0 / (window_size + 1.0)
        self.sigma_coef = sigma_coef
        self.robust_sogma_coef = robust_sogma_coef
        
        # величина запаздывания скользящего среднего
        self.offset = (self.window_size - 1)//2
        
        # Коэффициент масштабирования для согласования MAD и Sigma (для нормального распр.)
        self.k_mad = 1.4826
        
        # Состояния
        self.median_est = 0.0  # Оценка центра (аналог медианы)
        self.mad_est = 0.0     # Оценка разброса (аналог MAD)
        self.initialized = False

    def process_signal(self, mask: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Обрабатывает весь сигнал целиком, но эмулирует потоковую обработку.
        Отмечает аномальные отсчёты в `mask` (на месте); возвращает (mask, median_array, mad_array).
        Цикл вынесен в numba-функцию `_robust_segment` (результат совпадает с прежней Python-версией побитно).
        """
        median_array, mad_array = _robust_segment(mask, signal, self.alpha, self.sigma_coef,
                                                  self.robust_sogma_coef, self.k_mad, self.offset)
        self.median_est = median_array[-1]
        self.mad_est = mad_array[-1]
        self.initialized = True
        return mask, median_array, mad_array


@njit(cache=True)
def _robust_segment(mask, signal, alpha, sigma_coef, robust_sigma_coef, k_mad, offset):
    """Цикл `OnlineRobustSegmenter.process_signal`: EMA-медиана и EMA-MAD с винзоризацией.

    Отсчёт i − offset помечается в `mask`, если отклоняется от оценки медианы больше чем на
    sigma_coef · k_mad · MAD (оценки по предыдущему состоянию — каузально). Возвращает истории оценок
    медианы и MAD (длина n − offset), как списки прежней версии.
    """
    n = len(signal)
    m = max(n - 1 - offset, 0) + 1
    median_array = np.empty(m, dtype=np.float64)
    mad_array = np.empty(m, dtype=np.float64)
    k = 0
    median_est = float(signal[0])
    mad_est = 1.0
    for i in range(1, n):
        if i > offset:
            median_array[k] = median_est
            mad_array[k] = mad_est
            k += 1
        # порог по ПРЕДЫДУЩЕМУ состоянию (каузальность)
        sigma = k_mad * mad_est
        threshold = sigma_coef * sigma
        if threshold < 1e-5:
            threshold = 1e-5
        if i > offset:
            deviation = signal[i - offset] - median_est
            if abs(deviation) > threshold:
                mask[i - offset] = 1
        # обновление статистик: выброс тянет оценку только до границы порога
        threshold = robust_sigma_coef * sigma
        if threshold < 1e-5:
            threshold = 1e-5
        x = signal[i]
        deviation = x - median_est
        if abs(deviation) > threshold:
            clipped_val = median_est + np.sign(deviation) * threshold
            median_est = alpha * clipped_val + (1 - alpha) * median_est
            input_for_mad_ema = threshold
        else:
            median_est = alpha * x + (1 - alpha) * median_est
            input_for_mad_ema = abs(deviation)
        mad_est = alpha * input_for_mad_ema + (1 - alpha) * mad_est
    median_array[k] = median_est
    mad_array[k] = mad_est
    return median_array, mad_array


class SquareNoises(Module):

    def __init__(self):
        """
            Adds a square-wave noise (`new_smooth_noise`, model of real interference) to the signal.

            The same noise is added to every channel (the real interference is common to all channels).
            Suppression is no longer performed here: it is done by the separate ICA step of the
            preprocessing pipeline (`eddy_preprocess.ICASuppression`), so input and output shapes match.
            In the eddy pipeline the input is the standardised 15-channel signal (step 6 in TASKS.md).
        """
        super().__init__()

    def forward(self, signals: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, noise = new_smooth_noise(signals.size(1))
        return signals + torch.from_numpy(noise[None, :]).to(signals.dtype), bboxes, labels
