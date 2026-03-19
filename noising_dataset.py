import cv2
import numpy as np
import os
import glob
import random

# --- CONFIGURAZIONE ---
INPUT_DIR = "/Users/francescogranata/Documents/datasets/DIV2K_train_HR"
OUTPUT_DIR = "DIV2K_train_LR_custom"
SCALE_FACTOR = 4
SEED = 42  # seed fisso

# --- PARAMETRI RANDOM (RANGES) ---
BLUR_KERNEL_CHOICES = [3, 5, 7]
BLUR_SIGMA_RANGE = (0.5, 10)
NOISE_RANGE = (5, 25)
JPEG_RANGE = (60, 95)

def set_seed(seed):
    """
    Fissa i seed per garantire la riproducibilità
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seed impostato a: {seed}")

def apply_random_degradation(img):
    """
    Applica la pipeline: Blur -> Resize -> Noise -> JPEG
    """
    # 1. BLUR (Sfocatura Gaussiana)
    kernel_size = np.random.choice(BLUR_KERNEL_CHOICES)
    kernel_size = (kernel_size, kernel_size)
    sigma = np.random.uniform(BLUR_SIGMA_RANGE[0], BLUR_SIGMA_RANGE[1])
    img_blur = cv2.GaussianBlur(img, kernel_size, sigma)
    
    # 2. DOWNSAMPLE (Ridimensionamento)
    h, w, _ = img_blur.shape
    new_h, new_w = h // SCALE_FACTOR, w // SCALE_FACTOR
    img_resized = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. NOISE (Rumore Gaussiano)
    noise_level = np.random.uniform(NOISE_RANGE[0], NOISE_RANGE[1])
    noise = np.random.normal(0, noise_level, img_resized.shape)
    img_noise = img_resized + noise
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
    
    # 4. JPEG COMPRESSION
    jpeg_quality = np.random.randint(JPEG_RANGE[0], JPEG_RANGE[1])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', img_noise, encode_param)
    img_noise = cv2.imdecode(encimg, 1)

    # 5. UPSAMPLE (Back to original size) - AGGIUNTA FONDAMENTALE
    # Riportiamo l'immagine alle dimensioni originali (o quasi) per darla in pasto a Uformer
    # Nota: Uformer lavora bene con dimensioni divisibili per 128 o la window size (8x8)
    img_final = cv2.resize(img_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return img_final

def main():
    set_seed(SEED)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Creata cartella: {OUTPUT_DIR}")

    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    
    if len(images) == 0:
        print(f"Errore: Nessuna immagine trovata in {INPUT_DIR}")
        return

    print(f"Trovate {len(images)} immagini. Inizio elaborazione...")

    count = 800 # Modifica questo valore per processare l'intero dataset

    for i, img_path in enumerate(images):
        if i >= count:
            break

        img = cv2.imread(img_path)
        if img is None:
            print(f"Impossibile leggere: {img_path}")
            continue
            
        img_lr = apply_random_degradation(img)
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        cv2.imwrite(save_path, img_lr)
        
        if (i + 1) % 50 == 0:
            print(f"Processate {i + 1}/{len(images)}...")

    print("Finito! Dataset creato")

if __name__ == "__main__":
    main()