import os
import cv2
import torch
import numpy as np
from basicsr.models.archs.NAFSSR_arch import NAFSSR
from basicsr.metrics import calculate_psnr

MODEL_PATH = './experiments/pretrained_models/NAFSSR-L_4x.pth'
INPUT_DIR = './result_baseline_noise'  # Prende l'input dal denoising
GT_DIR = './datasets/DIV2K_valid_HR'
SAVE_DIR = './result_baseline_N_SR'


OFFSET = 74

TILE_SIZE = 256
TILE_PAD = 16
SCALE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizzando il dispositivo: {device}")

def tiled_inference(model, img_in_stereo, tile_size, tile_pad, scale):
    """Esegue l'inferenza dividendo l'immagine in tasselli per risparmiare memoria video"""
    b, c, h, w = img_in_stereo.shape
    output = torch.zeros((b, 3, h * scale, w * scale)).to(device)
    stride = tile_size - tile_pad * 2

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)

            tile = img_in_stereo[:, :, y1:y2, x1:x2]

            with torch.no_grad():
                tile_out = model(tile)
                if isinstance(tile_out, list): tile_out = tile_out[0]
                tile_out = tile_out[:, :3, :, :]

            output[:, :, y1*scale:y2*scale, x1*scale:x2*scale] = tile_out

    return output

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = NAFSSR(up_scale=4, width=128, num_blks=128).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    model.eval()

    full_img_list = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg'))])
    total_images = len(full_img_list)

    img_list = full_img_list[OFFSET-1:]

    print(f"Immagini totali rilevate: {total_images}")
    print(f"Ripresa dall'immagine numero {OFFSET}. Rimanenti: {len(img_list)}")

    psnrs = []

    for i, filename in enumerate(img_list):
        current_idx = i + OFFSET
        img_lq = cv2.imread(os.path.join(INPUT_DIR, filename))
        if img_lq is None: continue

        img_in = torch.from_numpy(img_lq).permute(2, 0, 1).float().divide(255.).unsqueeze(0).to(device)
        img_in_stereo = torch.cat([img_in, img_in], dim=1)

        print(f"[{current_idx}/{total_images}] Processando {filename}...")

        try:
            output_tensor = tiled_inference(model, img_in_stereo, TILE_SIZE, TILE_PAD, SCALE)

            output = output_tensor.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            output = (output * 255.0).round().astype(np.uint8)

            img_gt = cv2.imread(os.path.join(GT_DIR, filename))
            if img_gt is not None:
                if output.shape != img_gt.shape:
                    output = cv2.resize(output, (img_gt.shape[1], img_gt.shape[0]))
                p = calculate_psnr(output, img_gt, crop_border=4)
                psnrs.append(p)
                print(f"PSNR: {p:.2f} dB")

            cv2.imwrite(os.path.join(SAVE_DIR, filename), output)

        except RuntimeError as e:
            print(f"Errore critico su {filename}: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            continue

    if psnrs:
        print(f"\nSessione conclusa. PSNR Medio: {np.mean(psnrs):.2f} dB")

if __name__ == '__main__':
    main()