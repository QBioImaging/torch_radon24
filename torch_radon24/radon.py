import torch
from .utils import fourier_filter, get_pad_width


class Radon(torch.nn.Module):
    """
    Radon Transformation

    Args:
        thetas (int): list of angles for radon transformation (default: [0, np.pi])
        image_size (int): edge length of input image (default: 400)
        circle (bool): if True, only the circle is reconstructed (default: False)
        filter_name (str): filter for backprojection, can be "ramp" or "shepp_logan" or "cosine" or "hamming" or "hann" (default: "ramp")
        device: (str): device can be either "cuda" or "cpu" (default: cuda)

    """

    def __init__(self, thetas=[0, torch.pi], circle=False, filter_name="ramp", device="cuda"):
        super(Radon, self).__init__()
        self.n_angles = len(thetas)
        self.circle = circle
        self.filter_name = filter_name
        self.device = device

        # get angles
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
        if self.circle == False:
            pad_width = get_pad_width(image_size)
            image_size = int(torch.ceil(torch.tensor((2**0.5) * image_size)))
            new_img = torch.nn.functional.pad(
                image,
                pad=[pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]],
                mode="constant",
                value=0,
            )
        else:
            new_img = image

        # calculate rotations
        rotations = torch.stack(
            (self.cos_al, self.sin_al, self.zeros, -self.sin_al, self.cos_al, self.zeros), -1
        ).reshape(-1, 2, 3)
        self.rotated = torch.nn.functional.affine_grid(
            rotations, torch.Size([self.n_angles, 1, image_size, image_size]), align_corners=True
        ).reshape(1, -1, image_size, 2)
        self.rotated = self.rotated.to(self.device)

        out_fl = torch.nn.functional.grid_sample(new_img, self.rotated.repeat(batch_size, 1, 1, 1), align_corners=True)
        out_fl = out_fl.reshape(batch_size, 1, self.n_angles, image_size, image_size)
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
        # get rotated grid
        tgrid = (grid_x * self.cos_al - grid_y * self.sin_al).unsqueeze(-1)
        y = torch.ones_like(tgrid) * torch.linspace(-1, 1, self.n_angles)[:, None, None, None]
        grid = torch.cat((y, tgrid), dim=-1).view(self.n_angles * det_count, det_count, 2)[None].to(self.device)
        grid = grid.repeat(bsz, 1, 1, 1)

        reconstruction_circle = (grid_x**2 + grid_y**2) <= 1
        reconstructed_circle = reconstruction_circle.repeat(bsz, 1, 1, 1)

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
        # Reconstruct
        reconstructed = torch.nn.functional.grid_sample(
            radon_filtered, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        reconstructed = reconstructed.view(bsz, self.n_angles, 1, det_count, det_count).sum(1)

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
