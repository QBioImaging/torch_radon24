import torch


def fourier_filter(name, size, device="cpu"):
    n = torch.cat((torch.arange(1, size / 2 + 1, 2, dtype=int), torch.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = torch.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2
    fourier_filter = 2 * torch.real(torch.fft.fft(f))

    if name == "ramp":
        pass
    elif name == "shepp_logan":
        omega = torch.pi * torch.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega

    elif name == "cosine":
        freq = torch.linspace(0, torch.pi - (torch.pi / size), size)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter

    elif name == "hamming":
        fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, periodic=False))

    elif name == "hann":
        fourier_filter *= torch.fft.fftshift(torch.hann_window(size, periodic=False))
    else:
        print(f"[TorchRadon] Error, unknown filter type '{name}', available filters are: 'ramp', 'shepp_logan', 'cosine', 'hamming', 'hann'")

    filter = fourier_filter.to(device)

    return filter



