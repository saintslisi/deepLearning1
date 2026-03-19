import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import glob
from tqdm import tqdm
import argparse

# --- PATH SETUP ---
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '..'))
sys.path.append(os.path.join(dir_name, '../../dataset'))

import utils
import options

# --- FUNZIONE PER ADATTARE LE DIMENSIONI (NO TILING) ---
def pad_to_multiple(input_tensor, multiple=8):
    """
    Uformer richiede che altezza e larghezza siano divisibili per 8 (window_size).
    Questa funzione aggiunge un bordino minuscolo se necessario, ma NON taglia l'immagine.
    """
    _, _, h, w = input_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h == 0 and pad_w == 0:
        return input_tensor, 0, 0
    
    # Padding reflect per non introdurre bordi neri
    input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return input_padded, pad_h, pad_w

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description='Test Full Frame 10 Images')
    opt = options.Options().init(parser).parse_args()
    
    # Parametri Modello Uformer_B
    opt.embed_dim = 32
    opt.depths = [1, 2, 8, 8]
    opt.win_size = 8
    opt.token_projection = 'linear'
    opt.token_mlp = 'leff'
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- PERCORSI (Modifica se necessario) ---
    val_input_dir = "../../dataset/test/input"     # Le tue immagini rumorose
    val_gt_dir = "../../dataset/test/groundtruth"  # Le tue immagini originali (per calcolo PSNR)
    model_path = "./logs/denoising/div2k_custom/Uformer_B_100epochs/models/model_best.pth" # IL TUO MODELLO
    result_dir = "./results/final_images_100epochs"
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 1. Carica Modello
    print(f"Loading model from: {model_path}")
    model = utils.get_arch(opt)
    
    # Caricamento robusto dei pesi
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 2. Lista Immagini (SOLO LE PRIME 10)
    files = sorted(glob.glob(os.path.join(val_input_dir, "*.png")))
    num_files = 10
    files_to_test = files[:num_files] # <--- Prendo solo le prime 10
    
    print(f"⚠️ ATTENZIONE: Test Full-Frame su {len(files_to_test)} immagini.")
    print("Se il Mac va in crash, significa che l'immagine intera occupa troppa RAM.")

    total_psnr = 0
    count = 0

    # 3. Loop di Test
    for file_path in tqdm(files_to_test):
        img_name = os.path.basename(file_path)
        
        # Leggi immagine
        img_input = cv2.imread(file_path)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        
        # Prepara Tensore
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_input)).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(device)
        
        # Adatta dimensioni (multiplo di 8)
        img_padded, pad_h, pad_w = pad_to_multiple(img_tensor, multiple=8)
        
        # INFERENZA FULL FRAME (Senza Tiling)
        try:
            with torch.no_grad():
                output_padded = model(img_padded)
        except RuntimeError as e:
            print(f"\n❌ ERRORE MEMORIA su {img_name}: {e}")
            print("L'immagine è troppo grande per la GPU. Salto...")
            continue

        # Rimuovi padding extra se necessario
        if pad_h > 0 or pad_w > 0:
            output_tensor = output_padded[:, :, :output_padded.shape[2]-pad_h, :output_padded.shape[3]-pad_w]
        else:
            output_tensor = output_padded

        # Converti e Salva
        restored_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_img = np.clip(restored_img * 255., 0, 255).astype(np.uint8)
        restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, img_name), restored_img_bgr)
        
        # Calcola PSNR
        gt_path = os.path.join(val_gt_dir, img_name)
        if os.path.exists(gt_path):
            img_gt = cv2.imread(gt_path)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            
            # Ridimensiona GT se serve (padding o differenze minime)
            if img_gt.shape != restored_img.shape:
                 img_gt = cv2.resize(img_gt, (restored_img.shape[1], restored_img.shape[0]))

            mse = np.mean((img_gt.astype(np.float32) - restored_img.astype(np.float32)) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else 100
            
            total_psnr += psnr
            count += 1

    print("-" * 30)
    if count > 0:
        print(f"Average PSNR (Full Frame) on {count} images: {total_psnr / count:.4f} dB")
    else:
        print("Nessuna immagine completata (Crash memoria?)")
    print(f"Images saved in: {result_dir}")
    print("-" * 30)

if __name__ == '__main__':
    main()