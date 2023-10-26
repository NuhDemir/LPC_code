import numpy as np
from scipy.signal import lfilter


# LPC katsayılarını hesaplayan fonksiyon
def compute_lpc_coefficients(signal, order):
    # Autokovaryans matrisini hesaplayın
    r = np.correlate(signal, signal, mode='full')
    r = r / np.max(r)

    # Levinson-Durbin rekürsif algoritma ile LPC katsayılarını hesaplayın
    a = np.zeros(order + 1)
    E = np.zeros(order + 1)
    E[0] = r[0]

    for i in range(1, order + 1):
        K = np.sum(a[1:i] * r[i - 1:0:-1])
        a[i] = (r[i] - K) / E[i - 1]
        E[i] = (1 - a[i] ** 2) * E[i - 1]

    return a[1:]


# LPC katsayılarını kullanarak sinyali yeniden oluşturun
def reconstruct_signal(lpc_coefficients, original_signal):
    lpc_coefficients = np.insert(-lpc_coefficients, 0, 1)
    synthesized_signal = lfilter(lpc_coefficients, 1, original_signal)
    return synthesized_signal


# Örnek ses sinyali
sample_signal = np.array([0.2, 0.7, 0.3, -0.5, -0.2, 0.3])

# LPC için sinyali analiz et
order = 4  # LPC katsayısı sırası
lpc_coefficients = compute_lpc_coefficients(sample_signal, order)

# LPC katsayılarını yazdır
print("LPC Katsayıları:", lpc_coefficients)

# LPC katsayılarını kullanarak sinyali yeniden oluşturun
reconstructed_signal = reconstruct_signal(lpc_coefficients, sample_signal)

# Orijinal sinyal ve yeniden oluşturulmuş sinyali karşılaştır
print("Orijinal Sinyal:", sample_signal)
print("Yeniden Oluşturulmuş Sinyal:", reconstructed_signal)
