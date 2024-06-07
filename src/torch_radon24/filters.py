import torch


def ramp_filter(size):
    n = torch.cat((torch.arange(1, size / 2 + 1, 2, dtype=int), torch.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = torch.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2
    return 2 * torch.real(torch.fft.fft(f))
