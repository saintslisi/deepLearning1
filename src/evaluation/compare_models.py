import os
import matplotlib.pyplot as plt
from PIL import Image

def crea_griglia_uformer(folder_path, save_path="confronto_uformer.png", crop_coords=None):
    """
    Crea una griglia 2x2 per confrontare i risultati specifici di Uformer.
    """
    # I nomi dei file che devi mettere nella cartella
    files_to_load = [
        "original.png", "noised.png",
        "uformer_baseline.png", "uformer_finetuned.png" 
    ]
    
    titles = [
        "(A) Originale (GT)", "(B) Input Degradato",
        "(C) Uformer - Baseline", "(D) Uformer - Fine-Tuned (100 epoche)"
    ]

    ref_path = os.path.join(folder_path, "original.png")
    if not os.path.exists(ref_path):
        print(f"Errore: File di riferimento {ref_path} non trovato!")
        return
    
    img_ref = Image.open(ref_path)
    target_size = img_ref.size
    images = []

    for filename in files_to_load:
        path = os.path.join(folder_path, filename)
        if not os.path.exists(path):
            print(f"Errore: Manca il file {filename} nella cartella {folder_path}")
            return
        
        img = Image.open(path).convert("RGB")

        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)

        if crop_coords:
            x, y, w, h = crop_coords
            img = img.crop((x, y, x+w, y+h))
        
        images.append(img)

    # Griglia 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Griglia Uformer salvata in: {save_path}")
    plt.show()

if __name__ == "__main__":
    cartella_attuale = "."
    # Imposta le coordinate per prendere il dettaglio (es. pelo dello scoiattolo o un bordo netto)
    ritaglio = (250, 250, 150, 150) # Modifica queste coordinate in base all'immagine 808
    
    crea_griglia_uformer(folder_path="",
                         save_path="confronto_uformer_crop.png",
                         crop_coords=ritaglio)