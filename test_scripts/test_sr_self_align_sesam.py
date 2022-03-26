import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.models.archs.rrdbnet_dcn_sesam_arch import RRDBDCNNetSESAM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/SR_Self_Align_x4_sesam/models/net_g_110000.pth'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='datasets/data_ovarian/val_input',
        help='input test image folder')
    parser.add_argument(
        '--device', type=str, default='cuda', help='Options: cuda, cpu.')
    args = parser.parse_args()

    device = torch.device(args.device)

    # set up model
    model = RRDBDCNNetSESAM(
        num_in_ch=1, num_out_ch=1, num_feat=64, num_block=20, num_grow_ch=32, num_extract_block=10, deformable_groups=8)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs('results/RRDBDCNNetSESAM', exist_ok=True)
    for idx, path in enumerate(
            sorted(glob.glob(os.path.join(args.folder, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img[0:1,:,:]
        img = img.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            output = model(img)
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'results/RRDBDCNNetSESAM/{imgname}_SR_Self_Align.png', output)


if __name__ == '__main__':
    main()
