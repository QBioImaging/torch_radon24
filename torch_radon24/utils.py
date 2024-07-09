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


def get_pad_width(image_size):
    """
    Pads the input image to make it square and centered, with non-zero elements in the middle.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch_size, 1, W, W)

    Returns:
        padded_image (torch.Tensor): Padded image tensor
        pad_width (list): Amount of padding applied to each dimension
    """

    # Compute diagonal and padding sizes
    diagonal = (2**0.5) * image_size
    pad = [int(torch.ceil(torch.tensor(diagonal - s))) for s in (image_size, image_size)]

    # Compute new and old centers
    new_center = [(s + p) // 2 for s, p in zip((image_size, image_size), pad)]
    old_center = [s // 2 for s in (image_size, image_size)]

    # Compute padding before and after
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = tuple((pb, p - pb) for pb, p in zip(pad_before, pad))

    return pad_width
