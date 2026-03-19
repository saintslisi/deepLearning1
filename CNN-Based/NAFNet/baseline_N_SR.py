import os
import cv2
import torch
import numpy as np
from basicsr.models.archs.NAFSSR_arch import NAFSSR
from basicsr.metrics import calculate_psnr

# --- CONFIG ---
MODEL_PATH = './experiments/pretrained_models/NAFSSR-L_4x.pth'
LR_DIR = './result_baseline_noise'
GT_DIR = './datasets/DIV2K_valid_HR'
SAVE_DIR = './result_baseline_N_SR'

device = torch.device('cpu')

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    model = NAFSSR(up_scale=4, width=128, num_blks=128)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    model.to(device).eval()

    img_list = sorted(os.listdir(LR_DIR))
    psnrs = []

    print(f"Inizio test su {len(img_list)} immagini...")

    for _,filename in enumerate(img_list):
        print(f"{_}/{len(img_list)}")
        img_lq = cv2.imread(os.path.join(LR_DIR, filename))
        if img_lq is None: continue
        
        img_in = torch.from_numpy(img_lq).permute(2, 0, 1).float().divide(255.).unsqueeze(0)
        
        img_in_stereo = torch.cat([img_in, img_in], dim=1).to(device)

        with torch.no_grad():
            output_stereo = model(img_in_stereo)
            
            if isinstance(output_stereo, list):
                output = output_stereo[0]
            else:
                output = output_stereo[:, :3, :, :]
        
        # Torna in formato immagine BGR
        output = output.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)

        # Calcolo PSNR
        img_gt = cv2.imread(os.path.join(GT_DIR, filename))
        if img_gt is not None:
            if output.shape != img_gt.shape:
                output = cv2.resize(output, (img_gt.shape[1], img_gt.shape[0]))
            
            p = calculate_psnr(output, img_gt, crop_border=4)
            psnrs.append(p)
            print(f"{filename}: {p:.2f} dB")

        cv2.imwrite(os.path.join(SAVE_DIR, filename), output)

    if psnrs:
        print(f"\nPSNR Medio Baseline: {np.mean(psnrs):.2f} dB")

if __name__ == '__main__':
    main()