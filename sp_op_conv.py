import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BlurGrad(Function):

    @staticmethod
    def forward(ctx, x, gauss_kernel, padding, groups):
        ctx.gauss_kernel = gauss_kernel
        ctx.padding = padding
        ctx.groups = groups
        return x

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = F.pad(grad_out, ctx.padding, mode='constant')
        return F.conv2d(grad_in, ctx.gauss_kernel, stride=1, padding=0, groups=ctx.groups), None, None, None


class BlurGradResidual(Function):

    @staticmethod
    def forward(ctx, x, gauss_kernel, padding, groups, alpha):
        ctx.gauss_kernel = gauss_kernel
        ctx.padding = padding
        ctx.groups = groups
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_out):
        def viz(t):
            return (t - t.min()) / (t.max() - t.min() + 1e-12) * 255
        alpha = ctx.alpha
        #cv2.imwrite('grad_out.png', viz(grad_out.data[0][0].abs().cpu().numpy()))
        grad_pad = F.pad(grad_out, [2, 2, 2, 2], mode='reflect')
        alpha_r = (grad_out - grad_pad[..., 4:, 2:-2]).abs()
        alpha_d = (grad_out - grad_pad[..., 2:-2, 4:]).abs()
        threshold = min(alpha_d.mean() + 2*alpha_d.std(), alpha_r.mean() + 2*alpha_r.std())
        alpha_d = torch.where(alpha_d > threshold, torch.ones_like(grad_out), torch.zeros_like(grad_out))
        alpha_r = torch.where(alpha_r > threshold, torch.ones_like(grad_out), torch.zeros_like(grad_out))
        alpha = alpha_d * alpha_r
        alpha1 = 1 - alpha
        #cv2.imwrite('alpha.png', alpha.data[0][0].cpu().numpy()*255)
        grad_in = F.pad(grad_out, ctx.padding, mode='constant')
        grad_in = alpha1 * F.conv2d(grad_in, ctx.gauss_kernel, stride=1, padding=0, groups=ctx.groups) + alpha * grad_out
        return grad_in, None, None, None, None


class BlurGradConv(nn.Module):
    def __init__(self, gauss_kernel_size, conv, use_residual=False, alpha=0.5):
        super(BlurGradConv, self).__init__()
        assert(gauss_kernel_size in [2, 3, 5, 7])
        K = gauss_kernel_size
        self.K = K
        gauss_kernel_dict = {
            2: [1, 1],
            3: [1, 2, 1],
            5: [1, 4, 6, 4, 1],
            7: [1, 6, 15, 20, 15, 6, 1]
        }
        pad_dict = {
            2: [0, 1, 0, 1],
            3: [1, 1, 1, 1],
            5: [2, 2, 2, 2],
            7: [3, 3, 3, 3]
        }
        self.pad = pad_dict[K]
        self.gauss = torch.FloatTensor(gauss_kernel_dict[K]).view(K, 1)
        self.gauss = self.gauss.matmul(self.gauss.view(1, K))
        self.gauss.div_(self.gauss.sum())
        self.gauss = self.gauss.view(1, 1, K, K).repeat(
            conv.in_channels, 1, 1, 1)
        self.gauss = nn.Parameter(self.gauss).requires_grad_(False)
        self.conv = conv
        self.blur_groups = conv.in_channels
        self.weight = self.conv.weight
        self.use_residual = use_residual
        self.alpha = alpha

    def forward(self, x):
        if self.use_residual:
            x = BlurGradResidual.apply(x, self.gauss, self.pad, self.blur_groups, self.alpha)
        else:
            x = BlurGrad.apply(x, self.gauss, self.pad, self.blur_groups)
        x = self.conv(x)
        return x

    def __repr__(self):
        return self.conv.__repr__() + '\tblur_kernel_size={}'.format(self.K)


class InterpGrad(Function):

    @staticmethod
    def forward(ctx, x, kernel, stride, padding):
        ctx.kernel = kernel
        ctx.kernel_size = ctx.kernel.shape[2:]
        ctx.stride = stride
        ctx.padding = padding
        ctx.x = x
        return F.conv2d(x, kernel, None, stride, padding)

    @staticmethod
    def backward(ctx, grad_out):
        B, C_out = grad_out.shape[:2]
        B, C_in = ctx.x.shape[:2]
        grad_in = F.interpolate(
            grad_out, scale_factor=ctx.stride, mode='bilinear')
        kernel = ctx.kernel.permute(1, 0, 2, 3)
        kernel = torch.rot90(kernel, 2, [2, 3])
        grad_in = F.conv2d(grad_in, kernel, None, stride=1, padding=ctx.padding)
        grad_out = grad_out.view(B, C_out, -1)
        cache_x = F.unfold(ctx.x, kernel_size=ctx.kernel_size,
                           stride=ctx.stride, padding=ctx.padding)
        L = cache_x.shape[1]
        cache_x = cache_x.view(B, L, -1).permute(0, 2, 1)
        grad_w = torch.bmm(grad_out, cache_x).sum(dim=0).view(
            C_out, C_in, ctx.kernel_size[0], ctx.kernel_size[1])
        return grad_in, grad_w, None, None


class InterpGradConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(InterpGradConv, self).__init__()
        assert(stride > 1)
        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return InterpGrad.apply(x, self.weight, self.stride, self.padding)


class ShiftConv(nn.Module):
    def __init__(self, conv, mode):
        super(ShiftConv, self).__init__()
        assert(mode in ['expand', 'divide'])
        self.mode = mode
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding

    def forward(self, x):
        H, W = x.shape[2:]
        x0 = F.pad(x, [0, 1, 0, 1], mode='reflect')
        if self.mode == 'expand':
            ret = None
            for i in range(4):
                x = x0[..., i//2:i//2+H, i % 2:i % 2+W]
                x = F.conv2d(x, self.weight, self.bias,
                             self.stride, self.padding)
                if ret is None:
                    ret = x
                else:
                    ret += x
            return ret
        elif self.mode == 'divide':
            grid = self.weight.shape[0] // 4
            x = torch.cat([x0[:grid, :, :H, :W],
                           x0[grid:2*grid, :, :H, 1:W+1],
                           x0[2*grid:3*grid, :, 1:H+1, :W],
                           x0[3*grid:, :, 1:H+1, 1:W+1]], dim=0)
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class MaskNorm(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.mask = torch.FloatTensor([[1, 2], [2, 4]])
        return x

    @staticmethod
    def backward(ctx, grad_out):
        B, C, H, W = grad_out.shape
        mask = ctx.mask.to(grad_out.device)
        grad_in = grad_out.view(B, C, H//2, 2, W//2,
                                2).permute(0, 1, 2, 4, 3, 5)
        grad_in /= mask
        grad_in = grad_in.permute(
            0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        return grad_in


class MaskNormConv(nn.Module):
    def __init__(self, conv):
        super(MaskNormConv, self).__init__()
        assert(conv.kernel_size == (3, 3) and conv.stride == (2, 2))
        self.conv = conv

    def forward(self, x):
        x = MaskNorm.apply(x)
        x = self.conv(x)
        return x


def test_blur_grad_conv():
    conv = nn.Conv2d(3, 6, 3, 1, 1)
    model = BlurGradConv(gauss_kernel_size=2, conv=conv).cuda()
    model.train()
    input = torch.randn(2, 3, 8, 8).cuda()
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(model.gauss.requires_grad)
    print('end')


def test_interp_grad_conv():
    model = InterpGradConv(3, 6, 3, 2, 1)
    model.train()
    model_conv = nn.Conv2d(3, 6, 3, 2, 1, bias=False)
    model_conv.weight.data = model.weight.data.clone()
    input = torch.randn(2, 3, 8, 8)
    output = model(input)
    print(output.shape)
    loss = output.sum()
    loss.backward()
    output_conv = model_conv(input)
    loss = output_conv.sum()
    loss.backward()
    print(id(model.weight.data))
    print(id(model_conv.weight.data))
    print(model.weight.data.mean())
    print(model_conv.weight.data.mean())
    print(model.weight.grad.data.mean())
    print(model_conv.weight.grad.data.mean())
    print('end')


def test_shift_conv():
    conv = nn.Conv2d(3, 6, 3, 2, 1)
    model = ShiftConv(conv, mode='expand')
    inputs = torch.randn(2, 3, 8, 8)
    output = model(inputs)
    print(output.shape)
    model.mode = 'divide'
    output = model(inputs)
    print(output.shape)


def test_mask_norm_conv():
    conv = nn.Conv2d(3, 6, 3, 2, 1)
    model = MaskNormConv(conv)
    inputs = torch.randn(2, 3, 8, 8)
    output = model(inputs)
    print(output.shape)
    loss = output.sum()
    loss.backward()
    print('end')


if __name__ == "__main__":
    # test_blur_grad_conv()
    # test_interp_grad_conv()
    # test_shift_conv()
    test_mask_norm_conv()
