"""Пороговые уровни шума рельса и замена битых каналов (алгоритм Кузьмина и др., МАИС 2017, 2018, 2021).

Перенесено из `EC_preprocess/kuzmin_levels.py`. Отличие от оригинала — `Get_Probability_Distribution`:
векторизована и **округляет** отсчёты до ближайшего целого вместо `int(x)` (отбрасывания дробной части).
Для целых отсчётов АЦП результат тот же; для нецелых (после ICA, аугментаций, в float32) ячейка 0
перестаёт быть вдвое шире остальных (что согласуется с `A[0] / 2` в `Get_Sigma`), а ошибка округления
float32 (37 → 36.99998) больше не сдвигает отсчёт в соседнюю ячейку и порог — на ступень.
"""
from typing import List
import numpy as np
import math

class ClassThreshold(object):  # объявление класса порогов
    def __init__(self, thplus=0, thmins=0, badchl=0):
        self.thplus = thplus  # положительный порог # int
        self.thmins = thmins  # отрицательный порог # int
        self.badchl = badchl  # признак битого канала # bool


def Get_Probability_Distribution(EC, Aplus, Amins, n, j):
    """Гистограммы |x| положительных и отрицательных отсчётов канала j (x округляется до целого).

    Нулевые отсчёты учитываются в обеих гистограммах, как в оригинале.
    """
    x = np.rint(EC[:, j]).astype(np.int64)
    x = x[np.abs(x) <= n]
    plus = x[x >= 0]
    mins = -x[x <= 0]
    Aplus += np.bincount(plus, minlength=n + 1)[:n + 1]
    Amins += np.bincount(mins, minlength=n + 1)[:n + 1]
    maxplus = int(plus.max()) if plus.size else 0
    maxmins = int(mins.max()) if mins.size else 0
    return maxplus, maxmins, int(plus.size), int(mins.size)

def Get_Sigma(A: np.ndarray, finis:int, total:int):
    all = total - (A[0] / 2)
    sum = 0.0
    ctr = A[0] / 2
    sig = 0.0
    for k in range(1, finis + 1):
        ctr += A[k]
        sum += A[k] * k * k
        oldsig = sig
        if ctr > 0:
            sig = math.sqrt(sum / ctr)
        else:
            sig = 0
        if ctr > (0.3 * all) and (3 * sig) < k and (3 * oldsig) >= (k - 1):
            break
    return sig

def Get_Corrected_Level(level:int, A:np.ndarray, n:int):
    resultlm = 0
    newlevel = level

    finis = int(min(level, n))

    lm = 1

    while lm > 0:
        lm = 0
        ctr = 0
        for k in range(0, finis):
            if A[k] <= A[k + 1]:
                lm += 1
                resultlm += 1
            else:
                A[ctr] = A[k]
                ctr += 1
        A[ctr] = A[finis]
        finis = ctr
        newlevel = finis

    return (newlevel, resultlm)

def Get_Signal_Thresholds(
    EC,
    Threshold1:List[ClassThreshold],
    Threshold2:List[ClassThreshold],
    CorrectedThreshold1:List[ClassThreshold],
    CorrectedThreshold2:List[ClassThreshold],
):

    for j in range(0, 15):
        thr1 = ClassThreshold()
        thr2 = ClassThreshold()
        corthr1 = ClassThreshold()
        corthr2 = ClassThreshold()
        # ------- вычисление сдвига -------
        n = 1024 * 4
        # ======= вычисление сдвига =======
        # ------- вычисление распределения частот -------
        Aplus = np.zeros(n + 1)
        Amins = np.zeros(n + 1)
        maxplus, maxmins, totalplus, totalmins = Get_Probability_Distribution(EC, Aplus, Amins, n, j)
        # 
        # ======= вычисление распределения частот =======
        # ------- вычисление sigma -------
        sigplus = Get_Sigma(Aplus, maxplus, totalplus)
        sigmins = Get_Sigma(Amins, maxmins, totalmins)
        levelplus = int(3 * sigplus + 0.5)
        levelmins = int(3 * sigmins + 0.5)
        # ======= вычисление sigma =======

        # ------- сохранение графика распределения частот в файл -------
        TempAplus = np.zeros(n + 1)
        TempAmins = np.zeros(n + 1)
        TempAplus[:] = Aplus[:]
        TempAmins[:] = Amins[:]
        # ======= сохранение графика распределения частот в файл =======

        # ------- корректировка пороговых уровней -------
        corlevelplus, lmplus = Get_Corrected_Level(levelplus, Aplus, n)
        corlevelmins, lmmins = Get_Corrected_Level(levelmins, Amins, n)
        # ======= корректировка пороговых уровней =======
        thr1.thplus = +int(levelplus)
        thr1.thmins = -int(levelmins)

        thr2.thplus = +int(2 * levelplus)
        thr2.thmins = -int(2 * levelmins)

        corthr1.thplus = +int(corlevelplus)
        corthr1.thmins = -int(corlevelmins)

        corthr2.thplus = +int(2 * corlevelplus)
        corthr2.thmins = -int(2 * corlevelmins)

        # ------- выявление битого канала -------
        if (
            lmplus + lmmins <= 4
            and maxplus <= 2 * (levelplus + 1)
            and maxmins <= 2 * (levelmins + 1)
        ):
            thr1.badchl = 1
            thr2.badchl = 1
            corthr1.badchl = 1
            corthr2.badchl = 1
        else:
            thr1.badchl = 0
            thr2.badchl = 0
            corthr1.badchl = 0
            corthr2.badchl = 0
        # ======= выявление битого канала =======

        Threshold1.append(thr1)
        Threshold2.append(thr2)
        CorrectedThreshold1.append(corthr1)
        CorrectedThreshold2.append(corthr2)
        
        
        
def find_closest(Threshold:List[ClassThreshold]):
    """Функция находит каналу ближайшую замену"""
    substitute:List[int] = [] # индекс ближайшего хорошего канала
    for line in range(0, 15):
        # Если канал хороший, ничего не заменяем
        if Threshold[line].badchl == 0:
            substitute.append(line)
            continue
        substitute.append(-1)
        #иначе ищем ближайший хороший канал
        for offset in range(1, 15):
            if line - offset >= 0:
                if Threshold[line-offset].badchl == 0:
                    substitute[line] = line - offset
                    break
            if line + offset < 15:
                if Threshold[line + offset].badchl == 0:
                    substitute[line] = line + offset
                    break 
    return substitute

def New_Channel_Substitution(
    EC:np.ndarray, Threshold1:List[ClassThreshold], Threshold2:List[ClassThreshold],
    CorrectedThreshold1:List[ClassThreshold], CorrectedThreshold2:List[ClassThreshold]
):
    substitute = find_closest(Threshold1)
    
    for line in range(0, 15):
        if substitute[line] != line and substitute[line] != -1:
            # print("КАНАЛ №", line, " БИТЫЙ!!!", "Заменяем на канал №", substitute[line])

            Threshold1[line].thplus = Threshold1[substitute[line]].thplus
            Threshold1[line].thmins = Threshold1[substitute[line]].thmins
            Threshold1[line].badchl = Threshold1[substitute[line]].badchl

            Threshold2[line].thplus = Threshold2[substitute[line]].thplus
            Threshold2[line].thmins = Threshold2[substitute[line]].thmins
            Threshold2[line].badchl = Threshold2[substitute[line]].badchl

            CorrectedThreshold1[line].thplus = CorrectedThreshold1[substitute[line]].thplus
            CorrectedThreshold1[line].thmins = CorrectedThreshold1[substitute[line]].thmins
            CorrectedThreshold1[line].badchl = CorrectedThreshold1[substitute[line]].badchl

            CorrectedThreshold2[line].thplus = CorrectedThreshold2[substitute[line]].thplus
            CorrectedThreshold2[line].thmins = CorrectedThreshold2[substitute[line]].thmins
            CorrectedThreshold2[line].badchl = CorrectedThreshold2[substitute[line]].badchl

            EC[:,line] = EC[:,substitute[line]]

# ======= замена битого канала на соседний =======