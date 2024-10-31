import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

from utils.inpainting_utils import *

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# dtype = torch.cuda.FloatTensor

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
dtype = torch.FloatTensor


def closure():
    global i

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()
    # 输出每一步的损失数值
    print('Iteration %05d    Loss %f' % (i, total_loss.item()))
    # 每隔20步保存图片
    if i % 20 == 0:
        np_to_pil(torch_to_np(out)).save(r"results/3/{}.jpg".format(i))
    i += 1

    return total_loss


for imgname in range(1, 4):
    imsize = -1
    dim_div_by = 32

    img_path = 'data/inpainting/ori/{}.jpg'.format(imgname)
    # 此处mask黑色表示需要修复的地方
    mask_path = 'data/inpainting/ori/mask{}_reverse.jpg'.format(imgname)

    NET_TYPE = 'skip_depth4'

    img_pil, img_np = get_image(img_path, imsize)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)

    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil = crop_image(img_pil, dim_div_by)
    img_pil.save(r"data/inpainting/crop/{}.jpg".format(imgname))
    img_mask_pil.save(r"data/inpainting/crop/mask{}_reverse.jpg".format(imgname))
    img_np = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)
    # 填充类型
    pad = 'reflection'
    OPT_OVER = 'net'
    # 优化器名称
    OPTIMIZER = 'adam'
    # 输入类型为随机向量
    INPUT = 'noise'

    input_depth = 1

    num_iter = 3001
    show_every = 50
    figsize = 8
    reg_noise_std = 0.00
    param_noise = True
    # 初始化网络
    if 'skip' in NET_TYPE:

        depth = int(NET_TYPE[-1])
        net = skip(input_depth, img_np.shape[0],
                   num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                   num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                   num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                   filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                   upsample_mode='nearest',  # downsample_mode='avg',
                   need1x1_up=False,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        LR = 0.01

    elif NET_TYPE == 'UNET':

        net = UNet(num_input_channels=input_depth, num_output_channels=3,
                   feature_scale=8, more_layers=1,
                   concat_x=False, upsample_mode='deconv',
                   pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

        LR = 0.001
        param_noise = False

    elif NET_TYPE == 'ResNet':

        net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')

        LR = 0.001
        param_noise = False

    else:
        assert False


    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
    # 网络参数量
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    i = 0
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(OPT_OVER, net, net_input)
    # 开始网络参数学习
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    # 最后图像保存
    out_np = torch_to_np(net(net_input))
    np_to_pil(out_np).save(r"results/{}/final.jpg".format(imgname))
