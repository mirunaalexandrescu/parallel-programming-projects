import cv2
import os
import random
import time
from copy import copy
import datetime


def load_images():
    """Carica le immagini di sfondo e primo piano"""
    # Carica sfondi
    backgrounds = []
    background_path = "input/background/"
    for filename in os.listdir(background_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(background_path + filename, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                backgrounds.append(img)

    # Carica primo piano
    foreground_path = "input/foreground/"
    foreground = None
    for filename in os.listdir(foreground_path):
        if filename.lower().endswith('.png'):
            foreground = cv2.imread(foreground_path + filename, cv2.IMREAD_UNCHANGED)
            if foreground is not None:
                foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
                break

    return backgrounds, foreground


def compose_images_sequential(foreground, backgrounds, num_images, save_images=True):
    """Composizione sequenziale delle immagini"""
    # Crea cartella output solo se necessario
    output_dir = None
    if save_images:
        timestamp = datetime.datetime.now()
        output_dir = f'output/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

    print(f"Inizio composizione sequenziale di {num_images} immagini...")

    for i in range(num_images):
        # Scegli uno sfondo casuale
        bg_index = random.randint(0, len(backgrounds) - 1)
        background = copy(backgrounds[bg_index])

        # Calcola posizione casuale per il foreground
        max_row = background.shape[0] - foreground.shape[0]
        max_col = background.shape[1] - foreground.shape[1]

        if max_row <= 0 or max_col <= 0:
            print("ERRORE: il foreground è più grande del background")
            continue

        row = random.randint(0, max_row - 1)
        col = random.randint(0, max_col - 1)

        # Alpha blending
        alpha_blend = random.randint(128, 255) / 255.0

        for j in range(foreground.shape[0]):
            for k in range(foreground.shape[1]):
                f_pixel = foreground[j, k]
                b_pixel = background[row + j, col + k]
                f_alpha = f_pixel[3] / 255.0

                if f_alpha > 0.9:  # Solo se il pixel del foreground non è trasparente
                    for c in range(3):  # RGB channels
                        background[row + j, col + k, c] = (
                                b_pixel[c] * (1 - alpha_blend) +
                                f_pixel[c] * alpha_blend * f_alpha
                        )

        # Salva l'immagine solo se richiesto
        if save_images:
            cv2.imwrite(f"{output_dir}/composed_{i}.png", background)

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"Processate {i + 1}/{num_images} immagini")

    print("Composizione sequenziale completata")
    return output_dir


def main():
    # Parametri di test
    test_sizes = [50, 100, 500]

    # Carica le immagini
    backgrounds, foreground = load_images()

    if not backgrounds or foreground is None:
        print("ERRORE: Impossibile caricare le immagini!")
        return

    print(f"Caricate {len(backgrounds)} immagini di sfondo")
    print(f"Caricato foreground di dimensioni: {foreground.shape}")

    # Test con diverse dimensioni
    for size in test_sizes:
        print(f"\n--- Test con {size} immagini ---")

        start_time = time.time()
        output_dir = compose_images_sequential(foreground, backgrounds, size)
        end_time = time.time()

        execution_time = end_time - start_time
        throughput = size / execution_time

        print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
        print(f"Throughput: {throughput:.2f} immagini/secondo")
        print(f"Immagini salvate in: {output_dir}")


if __name__ == "__main__":
    main()