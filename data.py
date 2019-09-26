import torch
import math
import cv2


def get_color_patch(patch_size=64):
    color = (
        (255, 0, 0),
        (255, 140, 0),
        (255, 215, 0),
        (0, 191, 255),
    )
    patchs = []
    for item in color:
        patch = torch.ones(patch_size, patch_size, 3)
        patch *= torch.FloatTensor(item).view(1, 1, 3)
        patchs.append(patch)
    result = []
    T = int(math.sqrt(len(color)))
    for i in range(T):
        tmp = []
        for j in range(T):
            tmp.append(patchs[i*T+j])
        result.append(torch.cat(tmp, dim=1))
    result = torch.cat(result, dim=0)
    return result


def get_color(patch_size=112):
    color = (
        (255, 140, 0)
    )
    patch = torch.ones(patch_size, patch_size, 3)
    patch *= torch.FloatTensor((255, 140, 0)).view(1, 1, 3)
    return patch


def test_get_color_patch():
    patch = get_color_patch().numpy()
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    cv2.imwrite('result.png', patch)


if __name__ == "__main__":
    test_get_color_patch()
