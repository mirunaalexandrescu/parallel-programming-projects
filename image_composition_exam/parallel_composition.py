import cv2
import os
import random
import time
from copy import copy
import datetime
import multiprocessing
from multiprocessing import Pool, Process
from joblib import Parallel, delayed
import math


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


def compose_single_image(foreground, background_img, output_dir, image_id, save_images=True):
    """Compone una singola immagine (per joblib)"""
    background = copy(background_img)

    # Posizione casuale
    max_row = background.shape[0] - foreground.shape[0]
    max_col = background.shape[1] - foreground.shape[1]

    if max_row <= 0 or max_col <= 0:
        return

    row = random.randint(0, max_row - 1)
    col = random.randint(0, max_col - 1)

    # Alpha blending
    alpha_blend = random.randint(128, 255) / 255.0

    for j in range(foreground.shape[0]):
        for k in range(foreground.shape[1]):
            f_pixel = foreground[j, k]
            b_pixel = background[row + j, col + k]
            f_alpha = f_pixel[3] / 255.0

            if f_alpha > 0.9:
                for c in range(3):
                    background[row + j, col + k, c] = (
                            b_pixel[c] * (1 - alpha_blend) +
                            f_pixel[c] * alpha_blend * f_alpha
                    )

    # Salva solo se richiesto
    if save_images and output_dir:
        cv2.imwrite(f"{output_dir}/composed_{image_id}.png", background)


def compose_batch_pool(foreground, backgrounds, output_dir, num_images, save_images=True):
    """Compone un batch di immagini (per Pool)"""
    for i in range(num_images):
        bg_index = random.randint(0, len(backgrounds) - 1)
        background = copy(backgrounds[bg_index])

        max_row = background.shape[0] - foreground.shape[0]
        max_col = background.shape[1] - foreground.shape[1]

        if max_row <= 0 or max_col <= 0:
            continue

        row = random.randint(0, max_row - 1)
        col = random.randint(0, max_col - 1)

        alpha_blend = random.randint(128, 255) / 255.0

        for j in range(foreground.shape[0]):
            for k in range(foreground.shape[1]):
                f_pixel = foreground[j, k]
                b_pixel = background[row + j, col + k]
                f_alpha = f_pixel[3] / 255.0

                if f_alpha > 0.9:
                    for c in range(3):
                        background[row + j, col + k, c] = (
                                b_pixel[c] * (1 - alpha_blend) +
                                f_pixel[c] * alpha_blend * f_alpha
                        )

        # Salva solo se richiesto
        if save_images and output_dir:
            process_name = multiprocessing.current_process().name
            cv2.imwrite(f"{output_dir}/composed_{process_name}_{i}.png", background)


def parallel_joblib(foreground, backgrounds, num_images, num_processes, save_images=True):
    """Parallelizzazione con joblib"""
    output_dir = None
    if save_images:
        timestamp = datetime.datetime.now()
        output_dir = f'output/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

    print(f"Inizio composizione parallela joblib ({num_processes} processi)")

    # Scegli un background per tutti
    bg_index = random.randint(0, len(backgrounds) - 1)
    background = backgrounds[bg_index]

    Parallel(n_jobs=num_processes)(
        delayed(compose_single_image)(foreground, background, output_dir, i, save_images)
        for i in range(num_images)
    )

    print("Composizione joblib completata")
    return output_dir


def parallel_pool(foreground, backgrounds, num_images, num_processes, save_images=True):
    """Parallelizzazione con Pool"""
    output_dir = None
    if save_images:
        timestamp = datetime.datetime.now()
        output_dir = f'output/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

    print(f"Inizio composizione parallela pool ({num_processes} processi)")

    images_per_process = math.ceil(num_images / num_processes)

    with Pool(processes=num_processes) as pool:
        args = [(foreground, backgrounds, output_dir, images_per_process, save_images)
                for _ in range(num_processes)]
        pool.starmap(compose_batch_pool, args)

    print("Composizione pool completata")
    return output_dir


def parallel_processes(foreground, backgrounds, num_images, num_processes, save_images=True):
    """Parallelizzazione con Process"""
    output_dir = None
    if save_images:
        timestamp = datetime.datetime.now()
        output_dir = f'output/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

    print(f"Inizio composizione parallela processes ({num_processes} processi)")

    images_per_process = math.ceil(num_images / num_processes)

    processes = []
    for i in range(num_processes):
        p = Process(target=compose_batch_pool,
                    args=(foreground, backgrounds, output_dir, images_per_process, save_images))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Composizione processes completata")
    return output_dir


def main():
    # Parametri
    test_sizes = [50, 100, 500]
    num_processes = 4

    # Carica immagini
    backgrounds, foreground = load_images()

    if not backgrounds or foreground is None:
        print("ERRORE: Impossibile caricare le immagini!")
        return

    print(f"Caricate {len(backgrounds)} immagini di sfondo")
    print(f"Usando {num_processes} processi")

    # Test per ogni dimensione
    for size in test_sizes:
        print(f"\n=== Test con {size} immagini ===")

        # Test Joblib
        start = time.time()
        joblib_dir = parallel_joblib(foreground, backgrounds, size, num_processes)
        joblib_time = time.time() - start
        print(f"Joblib: {joblib_time:.2f}s ({size / joblib_time:.1f} img/s)")

        # Test Pool
        start = time.time()
        pool_dir = parallel_pool(foreground, backgrounds, size, num_processes)
        pool_time = time.time() - start
        print(f"Pool: {pool_time:.2f}s ({size / pool_time:.1f} img/s)")

        # Test Processes
        start = time.time()
        proc_dir = parallel_processes(foreground, backgrounds, size, num_processes)
        proc_time = time.time() - start
        print(f"Processes: {proc_time:.2f}s ({size / proc_time:.1f} img/s)")


if __name__ == "__main__":
    main()