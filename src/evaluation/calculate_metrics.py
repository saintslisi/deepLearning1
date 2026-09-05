import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm
import glob
import argparse

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    parser = argparse.ArgumentParser(description='Calcola PSNR e SSIM su una cartella di ricostruzioni')
    parser.add_argument('--restored_dir', type=str, default='results/uformer',
                        help='cartella con le immagini ricostruite dal modello')
    parser.add_argument('--gt_dir', type=str, default='data/DIV2K_valid_HR',
                        help='cartella con le immagini originali')
    args = parser.parse_args()
    folder_restored, folder_gt = args.restored_dir, args.gt_dir

    files = sorted(glob.glob(os.path.join(folder_restored, "*.png")))
    print(f"Calcolo metriche su {len(files)} immagini...")

    avg_psnr = 0
    avg_ssim = 0
    count = 0

    for file_path in tqdm(files):
        img_name = os.path.basename(file_path)
        path_gt = os.path.join(folder_gt, img_name)

        if not os.path.exists(path_gt):
            print(f"GT mancante per {img_name}, salto.")
            continue

        # Carica immagini
        img_restored = cv2.imread(file_path)
        img_gt = cv2.imread(path_gt)

        # Controlla dimensioni (devono essere identiche)
        if img_restored.shape != img_gt.shape:
            # Se differiscono di poco (es. padding), ridimensiona il GT
            img_gt = cv2.resize(img_gt, (img_restored.shape[1], img_restored.shape[0]))

        # Calcola PSNR
        psnr_val = calculate_psnr(img_gt.astype(np.float32), img_restored.astype(np.float32))

        # Calcola SSIM (convertiamo in scala di grigi per la struttura)
        gray_restored = cv2.cvtColor(img_restored, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        
        ssim_val = ssim_func(gray_gt, gray_restored)

        avg_psnr += psnr_val
        avg_ssim += ssim_val
        count += 1

    print("-" * 30)
    print(f"RISULTATI FINALI su {count} immagini:")
    print(f"PSNR Medio: {avg_psnr / count:.4f} dB")
    print(f"SSIM Medio: {avg_ssim / count:.4f}")
    print("-" * 30)

    #salvo i risultati su file
    with open(os.path.join(folder_restored, "metrics_results_100epochs.txt"), "w") as f:
        f.write(f"RISULTATI FINALI su {count} immagini:\n")
        f.write(f"PSNR Medio: {avg_psnr / count:.4f} dB\n")
        f.write(f"SSIM Medio: {avg_ssim / count:.4f}\n")

if __name__ == '__main__':
    main()