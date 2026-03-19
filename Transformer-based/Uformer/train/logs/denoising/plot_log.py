import matplotlib.pyplot as plt
import re
import os

# Nome del tuo file di log
log_filename = './div2k_custom/Uformer_B_100epochs/results/2026-01-03_18-38-29.txt'

epochs = []
losses = []
psnrs = []

# Controllo se il file esiste
if not os.path.exists(log_filename):
    print(f"⚠️ Errore: Non trovo il file '{log_filename}' nella cartella corrente.")
    exit()

print(f"Lettura del file {log_filename} in corso...")

with open(log_filename, 'r') as f:
    for line in f:
        # 1. PARSING LOSS e EPOCH
        # Cerca righe tipo: "Epoch: 1 ... Loss: 3.9392 ..."
        if "Epoch:" in line and "Loss:" in line:
            try:
                parts = line.split()
                # Trova l'indice delle parole chiave per sicurezza
                ep_idx = parts.index("Epoch:")
                loss_idx = parts.index("Loss:")
                
                epoch_val = int(parts[ep_idx + 1])
                loss_val = float(parts[loss_idx + 1])
                
                epochs.append(epoch_val)
                losses.append(loss_val)
            except ValueError:
                continue

        # 2. PARSING PSNR
        # Cerca righe tipo: "[Ep 1 it 98 PSNR: 24.1923 ]"
        if "PSNR:" in line and "[Ep" in line:
            try:
                # Usa Regex per estrarre il primo numero dopo "PSNR:"
                match = re.search(r'PSNR:\s*([0-9.]+)', line)
                if match:
                    psnr_val = float(match.group(1))
                    psnrs.append(psnr_val)
            except ValueError:
                continue

# Controllo consistenza dati
min_len = min(len(epochs), len(losses), len(psnrs))
if min_len == 0:
    print("⚠️ Non sono riuscito a estrarre dati. Controlla il formato del file log.")
    exit()

# Allineiamo le liste (tagliamo eventuali dati extra finali se il log fosse interrotto a metà riga)
epochs = epochs[:min_len]
losses = losses[:min_len]
psnrs = psnrs[:min_len]

print(f"✅ Dati estratti: {min_len} epoche trovate.")
print(f"   Start PSNR: {psnrs[0]} dB -> End PSNR: {psnrs[-1]} dB")
print(f"   Start Loss: {losses[0]} -> End Loss: {losses[-1]}")

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Grafico 1: Training Loss
ax1.plot(epochs, losses, color='#E24A33', linewidth=2, label='Training Loss')
ax1.set_title('Training Loss (Charbonnier)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoche', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Grafico 2: Validation PSNR
ax2.plot(epochs, psnrs, color='#348ABD', linewidth=2, label='Validation PSNR')
ax2.set_title('Validation PSNR (Qualità)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoche', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# Annotazione sul valore massimo
max_psnr = max(psnrs)
max_epoch = epochs[psnrs.index(max_psnr)]
ax2.annotate(f'Best: {max_psnr:.2f} dB', 
             xy=(max_epoch, max_psnr), 
             xytext=(max_epoch-15, max_psnr-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_fine_100epochs.png', dpi=300)
print("✅ Grafico salvato come: plot_fine_100epochs.png")
plt.show()