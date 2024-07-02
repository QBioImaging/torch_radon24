import torch
import numpy as np
import matplotlib.pyplot as plt
from .filters import fourier_filter


class Radon(torch.nn.Module):
    """
    Radon Transformation

    Args:
        n_angles (int): number of projection angles for radon transformation (default: 1000)
        image_size (int): edge length of input image (default: 400)
        circle (bool): if True, only the circle is reconstructed (default: False)
        det_count (int): number of detectors (default: None)
        filter (str): filter for backprojection, can be "ramp" or "shepp_logan" or "cosine" or "hamming" or "hann" (default: "ramp")
        device: (str): device can be either "cuda" or "cpu" (default: cuda)

    """

    def __init__(self, n_angles=1000, image_size=400, circle=False, det_count=None, filter="ramp", device="cuda"):
        super(Radon, self).__init__()
        self.n_angles = n_angles
        self.image_size = image_size
        # get angles
        thetas = torch.linspace(0, np.pi, n_angles)[:, None, None]
        cos_al, sin_al = thetas.cos(), thetas.sin()
        zeros = torch.zeros_like(cos_al)
        # calculate rotations
        rotations = torch.stack((cos_al, sin_al, zeros, -sin_al, cos_al, zeros), -1).reshape(-1, 2, 3)
        self.rotated = torch.nn.functional.affine_grid(rotations, torch.Size([n_angles, 1, image_size, image_size]), align_corners=True).reshape(1, -1, image_size, 2)
        self.rotated = self.rotated.to(device)

        # init for back projection
        if det_count is None:
            det_count = image_size

        self.step_size = image_size / det_count
        self.circle = circle

        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, image_size), torch.linspace(-1, 1, image_size), indexing="ij")
        # get rotated grid
        tgrid = (grid_x * thetas.cos() - grid_y * thetas.sin()).unsqueeze(-1)
        y = torch.ones_like(tgrid) * torch.linspace(-1, 1, n_angles)[:, None, None, None]
        self.grid = torch.cat((y, tgrid), dim=-1).view(self.n_angles * self.image_size, self.image_size, 2)[None].to(device)
        self.reconstruction_circle = (grid_x**2 + grid_y**2) <= 1

        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(det_count)).float().log2().ceil()))
        self.pad_width = projection_size_padded - det_count
        if filter is None:
            self.filter = None
        else:
            self.filter = fourier_filter(name=filter, size=projection_size_padded, device=device)

    def forward(self, image):
        """Apply radon transformation on input image.

        Args:
            image (torch.tensor, (bzs, 1, W, H)): input image

        Returns:
            out (torch.tensor, (bzs, 1, W, angles)): sinogram
        """
        bsz, _, W, H = image.shape

        # Store original shape and padding/cropping info
        self.resize_info = {"original_shape": (W, H), "pad": False, "crop": False}

        if W != self.image_size or H != self.image_size:
            diff_W = self.image_size - W
            diff_H = self.image_size - H
            if diff_W > 0 or diff_H > 0:
                # Padding if the image is smaller than self.image_size
                pad_W = max(diff_W // 2, 0)
                pad_H = max(diff_H // 2, 0)
                image = torch.nn.functional.pad(image, (pad_H, pad_H, pad_W, pad_W), mode="constant", value=0)
                self.resize_info["pad"] = (pad_W, pad_H)
            else:
                # Center crop if the image is larger than self.image_size
                crop_W_start = (W - self.image_size) // 2
                crop_H_start = (H - self.image_size) // 2
                image = image[:, :, crop_W_start : crop_W_start + self.image_size, crop_H_start : crop_H_start + self.image_size]
                self.resize_info["crop"] = (crop_W_start, crop_H_start)

        out_fl = torch.nn.functional.grid_sample(image, self.rotated.repeat(bsz, 1, 1, 1), align_corners=True)
        out_fl = out_fl.reshape(bsz, 1, self.n_angles, self.image_size, self.image_size)
        out = out_fl.sum(3).permute(0, 1, 3, 2)
        return out

    def filter_backprojection(self, input):
        """Apply (filtered) backprojection on input sinogram.

        Args:
            input (torch.tensor, (bzs, 1, W, angles)): sinogram

        Returns:
            out (torch.tensor, (bzs, 1, W, H)): reconstructed image
        """
        bsz, _, det_count, _ = input.shape
        if self.filter is not None:
            # Pad input
            padded_input = torch.nn.functional.pad(input, [0, 0, 0, self.pad_width], mode="constant", value=0)
            # Apply filter
            projection = torch.fft.fft(padded_input, dim=2) * self.filter[:, None]
            radon_filtered = torch.real(torch.fft.ifft(projection, dim=2))[:, :, :det_count, :]
        else:
            radon_filtered = input
        # Reconstruct
        grid = self.grid.repeat(bsz, 1, 1, 1)
        reconstructed = torch.nn.functional.grid_sample(radon_filtered, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        reconstructed = reconstructed.view(bsz, self.n_angles, 1, self.image_size, self.image_size).sum(1)
        reconstructed = reconstructed / self.step_size
        # Circle
        if self.circle:
            reconstructed_circle = self.reconstruction_circle.repeat(bsz, 1, 1, 1)
            reconstructed[reconstructed_circle == 0] = 0.0

        reconstructed = reconstructed * np.pi / (2 * self.n_angles)

        # Resize the reconstructed image back to the original shape
        original_shape = self.resize_info["original_shape"]
        if self.resize_info["pad"]:
            pad_W, pad_H = self.resize_info["pad"]
            reconstructed = reconstructed[:, :, pad_W : pad_W + original_shape[0], pad_H : pad_H + original_shape[1]]
        elif self.resize_info["crop"]:
            # crop_W_start, crop_H_start = self.resize_info['crop']
            pad_W = (original_shape[0] - self.image_size) // 2
            pad_H = (original_shape[1] - self.image_size) // 2
            reconstructed = torch.nn.functional.pad(reconstructed, (pad_H, pad_H, pad_W, pad_W), mode="constant", value=0)

        return reconstructed
