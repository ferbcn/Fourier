import numpy as np
import matplotlib.pyplot as plt
import time


def calc_dft(data):
    """
    Compute the DISCRETE Fourier Transform of the 1D array x
    :param data: (array)
    """
    length = data.size
    n = np.arange(length)
    k = n.reshape((length, 1))
    e = np.exp(-2j * np.pi * k * n / length)
    return np.dot(e, data)


def calc_fft(data):
    """
    Compute the FAST Fourier Transform of the 1D array x
    :param data: (array)
    """
    return np.fft.fft(data)


if __name__ == "__main__":

    # generate some data (some sine waves at different frequencies)
    t = np.linspace(0, 1, 2000)
    s = np.sin(40 * 2 * np.pi * t) + 0.6 * np.sin(90 * 2 * np.pi * t) + 1.5 * np.sin(1000 * 2 * np.pi * t)

    # create figure and plots
    fig, axs = plt.subplots(5)
    fig.canvas.set_window_title('Fourier Transformations')
    fig.tight_layout()
    # fig.suptitle('DFT vs FFT')

    axs[0].set(xlabel='Time [s]', ylabel='Amplitude')
    axs[0].set_title('Time Domain')
    axs[0].plot(t, s)

    # Discrete Fourier Transformation (sooo slow)
    ini = time.time()
    dft = calc_dft(s)
    elapsed_dft = time.time() - ini
    print(f'DFT took: {elapsed_dft}ms')

    # Fast Fourier Transformation (FFT)
    ini = time.time()
    fft = calc_fft(s)
    elapsed_fft = time.time() - ini
    print(f'FFT took: {elapsed_fft}ms')

    # define spectrogram
    T = t[1] - t[0]  # sampling interval
    N = s.size

    # 1/T = frequency
    f = np.linspace(0, 1 / T, N)

    # draw plot
    axs[1].set(xlabel='Frequency [Hz]', ylabel='Amplitude')
    axs[1].set_title('Frequency Domain (DFT)')
    axs[1].bar(f[:N // 2], np.abs(dft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor

    # draw plot
    axs[2].set(xlabel='Frequency [Hz]', ylabel='Amplitude')
    axs[2].set_title('Frequency Domain (FFT)')
    axs[2].bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor

    # Computation Times
    x = ['FFT', 'DFT']
    y = [elapsed_fft, elapsed_dft]
    axs[3].set(xlabel='Time [Ms]', ylabel='Algorithm')
    axs[3].set_title('Computation Time DFT vs FFT')
    axs[3].barh(x, y)

    # compare data
    diffs = list()
    for i, point in enumerate(dft):
        diffs.append(dft[i] - fft[i])
    diffs = diffs[:N // 2]

    # draw plot
    axs[4].set(xlabel='Frequency [Hz]', ylabel='Amp. Difference')
    axs[4].set_title('Computation Difference')
    axs[4].bar(f[:N // 2], diffs, width=1.5)  # 1 / N is a normalization factor

    plt.show()
