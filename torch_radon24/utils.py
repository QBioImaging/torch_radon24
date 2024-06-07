def test_adjoint():
    """Tests if Radon operator and Backprojection operator are adjoint
    by running  <radon(x),y> / <x,fbp(y)>.
    """
    n_angles = 50
    image_size = 100
    device = "cpu"
    # load operators
    radon_op = radon(n_angles=n_angles, image_size=image_size, device=device)
    fbp_op = fbp(n_angles=n_angles, image_size=image_size, circle=True, device=device, filtered=True)
    # run operators on random tensors
    x = torch.rand([1, 1, image_size, image_size]).to(device)
    y = torch.rand([1, 1, image_size, n_angles]).to(device)
    leftside = torch.sum(radon_op(x) * y).item()
    rightside = torch.sum(x * fbp_op(y)).item()
    # print
    print("\n<Ax,y>=", leftside, "  -----  <x,A'y>=", rightside)
    print("\n leftside/rightside: ", leftside / rightside)
    return leftside / rightside
