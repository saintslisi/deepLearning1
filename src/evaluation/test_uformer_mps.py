import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import glob
from tqdm import tqdm
import argparse

# --- PATH SETUP ---

from src import utils
from src.training import options

# --- FUNZIONE DI INFERENZA A BLOCCHI (TILING) ---
def tiled_inference(model, input_tensor, tile_size=512, overlap=0):
    """
    Taglia l'immagine in blocchi, processa e rincolla.
    tile_size=512 è sicuro per Mac 16GB.
    """
    b, c, h, w = input_tensor.shape
    
    # 1. Padding per rendere l'immagine divisibile per tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    output_padded = torch.zeros_like(input_padded)
    
    H_pad, W_pad = input_padded.shape[2], input_padded.shape[3]

    # 2. Loop sui blocchi
    for y in range(0, H_pad, tile_size):
        for x in range(0, W_pad, tile_size):
            # Estrai il blocco
            input_tile = input_padded[:, :, y:y+tile_size, x:x+tile_size]
            
            # Processa il blocco
            with torch.no_grad():
                output_tile = model(input_tile)
            
            # Incolla il blocco
            output_padded[:, :, y:y+tile_size, x:x+tile_size] = output_tile

    # 3. Ritaglia via il padding extra
    return output_padded[:, :, :h, :w]

# --- MAIN ---
def main():
    # Configurazione manuale (Hardcoded per semplicità)
    parser = argparse.ArgumentParser(description='Test di Uformer con tiling')
    parser = options.Options().init(parser)
    parser.add_argument('--input_dir', type=str, default='data/DIV2K_valid_degraded',
                        help='cartella con le immagini degradate')
    parser.add_argument('--gt_dir', type=str, default='data/DIV2K_valid_HR',
                        help='cartella con le immagini di riferimento, per il PSNR')
    parser.add_argument('--result_dir', type=str, default='results/uformer',
                        help='cartella dove salvare le immagini ricostruite')
    parser.add_argument('--weights', type=str, required=True,
                        help='checkpoint del modello da valutare')
    opt = parser.parse_args()
    
    # Parametri Modello (devono essere IDENTICI al training)
    opt.embed_dim = 32
    opt.depths = [1, 2, 8, 8]
    opt.win_size = 8
    opt.token_projection = 'linear'
    opt.token_mlp = 'leff'
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    val_input_dir = opt.input_dir
    val_gt_dir = opt.gt_dir
    model_path = opt.weights
    result_dir = opt.result_dir
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 1. Carica Modello
    # print(f"Loading model from: {model_path}")
    # model = utils.get_arch(opt)
    # checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.to(device)
    # model.eval()

    # 1. Carica Modello
    print(f"Loading model from: {model_path}")
    model = utils.get_arch(opt)
    
    # --- FIX CARICAMENTO PESI (UNIVERSALE) ---
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # A. Gestione della chiave 'state_dict'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Il file è direttamente i pesi
        
    # B. Gestione del prefisso 'module.' (residuo di training multi-GPU)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Rimuoviamo 'module.' se presente all'inizio della chiave
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    # Carichiamo i pesi puliti
    model.load_state_dict(new_state_dict)
    # ------------------------------------------
    
    model.to(device)
    model.eval()

    # 2. Lista Immagini
    files = sorted(glob.glob(os.path.join(val_input_dir, "*.png")))
    print(f"Found {len(files)} images in test set.")

    total_psnr = 0
    count = 0

    # 3. Loop di Test
    for file_path in tqdm(files):
        # Leggi immagine
        img_name = os.path.basename(file_path)
        img_input = cv2.imread(file_path)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        
        # Prepara Tensore
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_input)).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(device)
        
        # INFERENZA (Usa la funzione Tiled per non crashare)
        restored_tensor = tiled_inference(model, img_tensor, tile_size=512)
        
        # Converti output in immagine
        restored_img = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_img = np.clip(restored_img * 255., 0, 255).astype(np.uint8)
        
        # Salva risultato
        restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, img_name), restored_img_bgr)
        
        # Calcola PSNR se esiste il Ground Truth
        gt_path = os.path.join(val_gt_dir, img_name)
        if os.path.exists(gt_path):
            img_gt = cv2.imread(gt_path)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            
            # Calcolo PSNR semplice
            mse = np.mean((img_gt.astype(np.float32) - restored_img.astype(np.float32)) ** 2)
            if mse == 0:
                psnr = 100
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            total_psnr += psnr
            count += 1

    print("-" * 30)
    print(f"Test Finished!")
    if count > 0:
        print(f"Average PSNR on {count} images: {total_psnr / count:.4f} dB")
    print(f"Images saved in: {result_dir}")
    print("-" * 30)

if __name__ == '__main__':
    main()