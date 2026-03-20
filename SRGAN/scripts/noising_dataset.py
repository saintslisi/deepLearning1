import cv2
import numpy as np
import os
import glob

# --- CONFIGURAZIONE ---
INPUT_DIR = "./data/DIV2K_test_HR"
OUTPUT_DIR = "./data/DIV2K_test_LR_noisy"
SCALE_FACTOR = 4                   # Di quanto riduciamo (x4 è standard)

# --- PARAMETRI RANDOM (RANGES) ---
# La sfocatura varierà tra leggera e media
BLUR_KERNEL_CHOICES = [3, 5, 7]     # Solo numeri dispari!
BLUR_SIGMA_RANGE = (0.5, 2.5)       # Da quasi nullo a sfocato

# Il rumore varierà da "pulito" a "molto rumoroso" (come la tua prova)
NOISE_RANGE = (5, 25)               # Min 5, Max 25

# La compressione varierà da "buona" a "artefatti visibili"
JPEG_RANGE = (60, 95)               # 95=Alta qualità, 60=Bassa qualità

def apply_random_degradation(img):
    """
    Applica la pipeline: Blur -> Resize -> Noise -> JPEG
    """
    
    # 1. BLUR (Sfocatura Gaussiana)
    # Simula una lente non perfettamente a fuoco
    kernel_size = np.random.choice(BLUR_KERNEL_CHOICES)
    kernel_size = (kernel_size, kernel_size)
    sigma = np.random.uniform(BLUR_SIGMA_RANGE[0], BLUR_SIGMA_RANGE[1])
    img_blur = cv2.GaussianBlur(img, kernel_size, sigma)
    
    # 2. DOWNSAMPLE (Ridimensionamento)
    # Riduciamo l'immagine di un fattore x4 usando l'interpolazione cubica
    h, w, _ = img_blur.shape
    new_h, new_w = h // SCALE_FACTOR, w // SCALE_FACTOR
    img_resized = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. NOISE (Rumore Gaussiano)
    # Aggiungiamo rumore casuale (simula ISO alti)
    noise_level = np.random.uniform(NOISE_RANGE[0], NOISE_RANGE[1])
    noise = np.random.normal(0, noise_level, img_resized.shape)
    img_noise = img_resized + noise
    
    # Clip per assicurarsi che i pixel rimangano tra 0 e 255 e conversione a uint8
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
    
    # 4. JPEG COMPRESSION
    # Simula gli artefatti di salvataggio
    # Codifichiamo in memoria e decodifichiamo subito
    jpeg_quality = np.random.randint(JPEG_RANGE[0], JPEG_RANGE[1])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', img_noise, encode_param)
    img_final = cv2.imdecode(encimg, 1)
    
    return img_final

def main():
    # Crea la cartella di output se non esiste
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Creata cartella: {OUTPUT_DIR}")

    # Trova tutte le immagini .png
    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    
    if len(images) == 0:
        print(f"Errore: Nessuna immagine trovata in {INPUT_DIR}")
        return

    print(f"Trovate {len(images)} immagini. Inizio elaborazione...")


    for i, img_path in enumerate(images):

        # Leggi immagine
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Impossibile leggere: {img_path}")
            continue
            
        # Applica degradazione
        img_lr = apply_random_degradation(img)
        
        # Salva immagine mantenendo lo stesso nome
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # Nota: Salviamo in PNG per non aggiungere ulteriore compressione non voluta
        # (gli artefatti JPEG sono già stati "impressi" nel passaggio 4)
        cv2.imwrite(save_path, img_lr)
        
        # Stampa progresso ogni 50 immagini
        if (i + 1) % 50 == 0:
            print(f"Processate {i + 1}/{len(images)}...")

    print("Finito! Dataset creato.")

if __name__ == "__main__":
    main()
