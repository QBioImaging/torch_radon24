import torch
from .utils import fourier_filter, get_pad_width

import torch.nn.functional as F


class Radon(torch.nn.Module):
    """
    Radon Transformation

    Args:
        thetas (int): list of angles for radon transformation (default: [0, np.pi])
        circle (bool): if True, only the circle is reconstructed (default: False)
        filter_name (str): filter for backprojection, can be "ramp" or "shepp_logan" or "cosine" or "hamming" or "hann" (default: "ramp")
        device: (str): device can be either "cuda" or "cpu" (default: cuda")
    """

    def __init__(self, thetas=[0, torch.pi], circle=False, filter_name="ramp", device="cuda"):
        super(Radon, self).__init__()
        self.n_angles = len(thetas)
        self.circle = circle
        self.filter_name = filter_name
        self.device = device

        # # get angles
        # thetas = torch.tensor(thetas, dtype=torch.float32)
        # self.cos_al, self.sin_al = thetas.cos(), thetas.sin()

        thetas = torch.tensor(thetas, dtype=torch.float32)[:, None, None]
        self.cos_al, self.sin_al = thetas.cos(), thetas.sin()
        self.zeros = torch.zeros_like(self.cos_al)

    def forward(self, image):
        """Apply radon transformation on input image.

        Args:
            image (torch.tensor, (bzs, 1, W, H)): input image

        Returns:
            out (torch.tensor, (bzs, 1, W, angles)): sinogram
        """
        batch_size, _, image_size, _ = image.shape
        # code for circle case
        if not self.circle:
            pad_width = get_pad_width(image_size)
            image_size = int(torch.ceil(torch.tensor((2**0.5) * image_size)))
            new_img = F.pad(
                image,
                pad=[pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]],
                mode="constant",
                value=0,
            )
        else:
            new_img = image

        # Calculate rotated images
        rotated_images = []
        for cos_al, sin_al in zip(self.cos_al, self.sin_al):
            theta = (
                torch.tensor([[cos_al, sin_al, 0], [-sin_al, cos_al, 0]], dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            grid = F.affine_grid(theta, torch.Size([1, 1, image_size, image_size]), align_corners=True)
            rotated_img = F.grid_sample(new_img, grid.repeat(batch_size, 1, 1, 1), align_corners=True)
            rotated_images.append(rotated_img)

        out_fl = torch.stack(rotated_images, dim=2).to(self.device)
        out = out_fl.sum(3).permute(0, 1, 3, 2)
        return out

    def filter_backprojection(self, sinogram):
        """Apply (filtered) backprojection on sinogram.

        Args:
            input (torch.tensor, (bzs, 1, W, angles)): sinogram

        Returns:
            out (torch.tensor, (bzs, 1, W, H)): reconstructed image
        """

        bsz, _, det_count, _ = sinogram.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, det_count), torch.linspace(-1, 1, det_count), indexing="ij"
        )
        reconstruction_circle = (grid_x**2 + grid_y**2) <= 1
        reconstructed_circle = reconstruction_circle.repeat(bsz, 1, 1, 1).to(self.device)

        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(det_count)).float().log2().ceil()))
        self.pad_width_sino = projection_size_padded - det_count

        if self.filter_name is not None:
            filter = fourier_filter(name=self.filter_name, size=projection_size_padded, device=self.device)
            # Pad input
            padded_input = torch.nn.functional.pad(sinogram, [0, 0, 0, self.pad_width_sino], mode="constant", value=0)
            # Apply filter
            projection = torch.fft.fft(padded_input, dim=2) * filter[:, None]
            radon_filtered = torch.real(torch.fft.ifft(projection, dim=2))[:, :, :det_count, :]
        else:
            radon_filtered = sinogram

        reconstructed = torch.zeros((bsz, 1, det_count, det_count), device=self.device)
        y_grid = torch.linspace(-1, 1, self.n_angles)

        # Reconstruct using a for loop
        for i, (cos_al, sin_al, y_colume) in enumerate(zip(self.cos_al, self.sin_al, y_grid)):
            tgrid = (grid_x * cos_al - grid_y * sin_al).unsqueeze(0).unsqueeze(-1)
            y = torch.ones_like(tgrid) * y_colume
            grid = torch.cat((y, tgrid), dim=-1).to(self.device)
            # Apply grid_sample to the current angle
            rotated_img = F.grid_sample(
                radon_filtered, grid.repeat(bsz, 1, 1, 1), mode="bilinear", padding_mode="zeros", align_corners=True
            )
            rotated_img = rotated_img.view(bsz, 1, det_count, det_count)
            # Sum the rotated images for backprojection
            reconstructed += rotated_img

        # Circle
        reconstructed[reconstructed_circle == 0] = 0.0
        reconstructed = reconstructed * torch.pi / (2 * self.n_angles)

        if self.circle == False:
            # center crop reconstructed to the output size = det_count / (2**0.5)
            output_size = int(torch.floor(torch.tensor(det_count / (2**0.5))))
            start_idx = (det_count - output_size) // 2
            end_idx = start_idx + output_size
            reconstructed = reconstructed[:, :, start_idx:end_idx, start_idx:end_idx]

        return reconstructed
