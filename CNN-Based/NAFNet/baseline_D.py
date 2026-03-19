import os
import cv2
import torch
import numpy as np
from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.metrics import calculate_psnr

MODEL_PATH = './experiments/pretrained_models/NAFNet-SIDD-width64.pth'
NOISY_DIR = './datasets/DIV2K_valid_LR_custom'
GT_DIR = './datasets/DIV2K_valid_HR'
SAVE_DIR = './result_baseline_D'

device = torch.device('cpu')

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    model = NAFNet(
        width=64, 
        enc_blk_nums=[2, 2, 4, 8], 
        middle_blk_num=12, 
        dec_blk_nums=[2, 2, 2, 2]
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device).eval()

    img_list = sorted(os.listdir(NOISY_DIR))
    psnrs = []

    print(f"Inizio Denoising su {len(img_list)} immagini...")

    for i, filename in enumerate(img_list):
        img_path = os.path.join(NOISY_DIR, filename)
        img_lq = cv2.imread(img_path)
        if img_lq is None: 
            continue
        
        img_in = torch.from_numpy(img_lq).permute(2, 0, 1).float().divide(255.).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_in)
        
        output = output.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)

        gt_path = os.path.join(GT_DIR, filename)
        if os.path.exists(gt_path):
            img_gt = cv2.imread(gt_path)
            if img_gt is not None:
                if output.shape != img_gt.shape:
                    output = cv2.resize(output, (img_gt.shape[1], img_gt.shape[0]))
                
                p = calculate_psnr(output, img_gt, crop_border=0)
                psnrs.append(p)
                print(f"[{i+1}/{len(img_list)}] {filename}: {p:.2f} dB")
        else:
            print(f"[{i+1}/{len(img_list)}] {filename}: Salvata (GT non trovata per PSNR)")

        cv2.imwrite(os.path.join(SAVE_DIR, filename), output)

    if psnrs:
        print(f"\nPSNR Medio Denoising (Baseline): {np.mean(psnrs):.2f} dB")
    print(f"Processo completato. Immagini salvate in: {SAVE_DIR}")

if __name__ == '__main__':
    main()