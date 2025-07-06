import time
import matplotlib.pyplot as plt
import multiprocessing
import csv
import datetime
import random
from sequential_composition import load_images, compose_images_sequential
from parallel_composition import parallel_joblib, parallel_pool, parallel_processes


def create_plots(results, num_processes):
    """Crea i grafici dei risultati"""
    sizes = results['sizes']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Grafico 1: Speedup vs Numero di Immagini
    joblib_speedup = [seq / par for seq, par in zip(results['sequential'], results['joblib'])]
    pool_speedup = [seq / par for seq, par in zip(results['sequential'], results['pool'])]
    proc_speedup = [seq / par for seq, par in zip(results['sequential'], results['processes'])]

    ax1.plot(sizes, joblib_speedup, 's-', label='Joblib', linewidth=2, markersize=6)
    ax1.plot(sizes, pool_speedup, '^-', label='Pool', linewidth=2, markersize=6)
    ax1.plot(sizes, proc_speedup, 'd-', label='Processes', linewidth=2, markersize=6)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax1.axhline(y=num_processes, color='green', linestyle='--', alpha=0.7,
                label=f'Speedup Ideale ({num_processes}x)')

    ax1.set_xlabel('Numero di Immagini')
    ax1.set_ylabel('Speedup')
    ax1.set_title(f'Speedup con {num_processes} Processi')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(joblib_speedup), max(pool_speedup), max(proc_speedup)) * 1.1)

    # Grafico 2: Tempi di esecuzione
    ax2.plot(sizes, results['sequential'], 'o-', label='Sequenziale', linewidth=2)
    ax2.plot(sizes, results['joblib'], 's-', label='Joblib', linewidth=2)
    ax2.plot(sizes, results['pool'], '^-', label='Pool', linewidth=2)
    ax2.plot(sizes, results['processes'], 'd-', label='Processes', linewidth=2)

    ax2.set_xlabel('Numero di Immagini')
    ax2.set_ylabel('Tempo (secondi)')
    ax2.set_title('Tempi di Esecuzione')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Grafico 3: Efficienza
    joblib_eff = [s / num_processes for s in joblib_speedup]
    pool_eff = [s / num_processes for s in pool_speedup]
    proc_eff = [s / num_processes for s in proc_speedup]

    ax3.plot(sizes, joblib_eff, 's-', label='Joblib', linewidth=2, markersize=6)
    ax3.plot(sizes, pool_eff, '^-', label='Pool', linewidth=2, markersize=6)
    ax3.plot(sizes, proc_eff, 'd-', label='Processes', linewidth=2, markersize=6)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Efficienza Ideale')

    ax3.set_xlabel('Numero di Immagini')
    ax3.set_ylabel('Efficienza')
    ax3.set_title('Efficienza Parallela')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150)
    plt.show()


def save_results_to_csv(results, num_processes):
    """Salva i risultati in un file CSV"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{num_processes}cores_{timestamp}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Header con info sistema
        writer.writerow(['# Benchmark Image Composition'])
        writer.writerow(['# Core utilizzati:', num_processes])
        writer.writerow(['# Data:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        # Header dati
        writer.writerow(['Num_Immagini', 'Metodo', 'Tempo_Secondi', 'Speedup', 'Throughput_ImgSec'])

        # Dati
        sizes = results['sizes']
        for i, size in enumerate(sizes):
            seq_time = results['sequential'][i]
            joblib_time = results['joblib'][i]
            pool_time = results['pool'][i]
            proc_time = results['processes'][i]

            # Riga sequenziale
            writer.writerow([size, 'Sequential', f"{seq_time:.3f}", "1.00", f"{size / seq_time:.1f}"])

            # Righe parallele
            writer.writerow(
                [size, 'Joblib', f"{joblib_time:.3f}", f"{seq_time / joblib_time:.2f}", f"{size / joblib_time:.1f}"])
            writer.writerow(
                [size, 'Pool', f"{pool_time:.3f}", f"{seq_time / pool_time:.2f}", f"{size / pool_time:.1f}"])
            writer.writerow(
                [size, 'Processes', f"{proc_time:.3f}", f"{seq_time / proc_time:.2f}", f"{size / proc_time:.1f}"])

    print(f"Risultati salvati in: {filename}")
    return filename


def print_summary(results, num_processes):
    """Stampa il riassunto dei risultati"""
    print(f"\n{'=' * 80}")
    print("RIASSUNTO RISULTATI BENCHMARK")
    print(f"{'=' * 80}")

    sizes = results['sizes']

    # TABELLA 1: Tempi di esecuzione
    print(f"\nTEMPI DI ESECUZIONE (secondi)")
    print("+" + "-" * 78 + "+")
    print(f"| {'Immagini':<10} | {'Sequenziale':<12} | {'Joblib':<12} | {'Pool':<12} | {'Processes':<12} |")
    print("+" + "-" * 78 + "+")

    for i, size in enumerate(sizes):
        seq_time = results['sequential'][i]
        joblib_time = results['joblib'][i]
        pool_time = results['pool'][i]
        proc_time = results['processes'][i]

        print(f"| {size:<10} | {seq_time:<12.2f} | {joblib_time:<12.2f} | {pool_time:<12.2f} | {proc_time:<12.2f} |")

    print("+" + "-" * 78 + "+")

    # TABELLA 2: Speedup
    print(f"\nSPEEDUP (vs Sequenziale)")
    print("+" + "-" * 66 + "+")
    print(f"| {'Immagini':<10} | {'Joblib':<12} | {'Pool':<12} | {'Processes':<12} | {'Migliore':<12} |")
    print("+" + "-" * 66 + "+")

    for i, size in enumerate(sizes):
        seq_time = results['sequential'][i]
        joblib_speedup = seq_time / results['joblib'][i]
        pool_speedup = seq_time / results['pool'][i]
        proc_speedup = seq_time / results['processes'][i]

        # Trova il migliore
        speedups = {'Joblib': joblib_speedup, 'Pool': pool_speedup, 'Processes': proc_speedup}
        best_method = max(speedups, key=speedups.get)
        best_speedup = speedups[best_method]

        print(
            f"| {size:<10} | {joblib_speedup:<12.2f} | {pool_speedup:<12.2f} | {proc_speedup:<12.2f} | {best_method:<8}{best_speedup:>3.2f}x |")

    print("+" + "-" * 66 + "+")

    # TABELLA 3: Throughput
    print(f"\nTHROUGHPUT (immagini/secondo)")
    print("+" + "-" * 78 + "+")
    print(f"| {'Immagini':<10} | {'Sequenziale':<12} | {'Joblib':<12} | {'Pool':<12} | {'Processes':<12} |")
    print("+" + "-" * 78 + "+")

    for i, size in enumerate(sizes):
        seq_throughput = size / results['sequential'][i]
        joblib_throughput = size / results['joblib'][i]
        pool_throughput = size / results['pool'][i]
        proc_throughput = size / results['processes'][i]

        print(
            f"| {size:<10} | {seq_throughput:<12.1f} | {joblib_throughput:<12.1f} | {pool_throughput:<12.1f} | {proc_throughput:<12.1f} |")

    print("+" + "-" * 78 + "+")

    # STATISTICHE FINALI
    print(f"\nSTATISTICHE RIASSUNTIVE")
    print("+" + "-" * 45 + "+")

    # Speedup medio
    avg_speedups = {
        'Joblib': sum(seq / par for seq, par in zip(results['sequential'], results['joblib'])) / len(sizes),
        'Pool': sum(seq / par for seq, par in zip(results['sequential'], results['pool'])) / len(sizes),
        'Processes': sum(seq / par for seq, par in zip(results['sequential'], results['processes'])) / len(sizes)
    }

    print(f"| {'Metodo':<15} | {'Speedup Medio':<12} | {'Efficienza':<12} |")
    print("+" + "-" * 45 + "+")

    for method, speedup in avg_speedups.items():
        efficiency = speedup / num_processes * 100
        print(f"| {method:<15} | {speedup:<12.2f} | {efficiency:<10.1f}% |")

    print("+" + "-" * 45 + "+")

    # Metodo migliore
    best_overall = max(avg_speedups, key=avg_speedups.get)
    best_speedup = avg_speedups[best_overall]

    print(f"\nCONFIGURAZIONE SISTEMA:")
    print(f"   Core utilizzati: {num_processes}")
    print(f"   Metodo con speedup maggiore: {best_overall} ({best_speedup:.2f}x)")
    print(f"   Efficienza massima: {best_speedup / num_processes:.1%}")


def run_benchmark(save_images=False):
    """Esegue il benchmark completo"""
    max_processes = multiprocessing.cpu_count()

    print(f"BENCHMARK COMPLETO")
    print("=" * 50)

    # Test da 50 a 500 immagini con step di 50
    test_sizes = range(50, 501, 50)

    print("Caricamento immagini...")
    backgrounds, foreground = load_images()

    if not backgrounds or foreground is None:
        print("ERRORE: Impossibile caricare le immagini!")
        return None, None, None

    print(f"Caricate {len(backgrounds)} immagini di sfondo")
    print(f"Usando {max_processes} processi")
    print(f"Salvataggio immagini: {'Abilitato' if save_images else 'Disabilitato'}")
    print(f"Test con: {list(test_sizes)} immagini")

    results = {
        'sizes': list(test_sizes),
        'sequential': [],
        'joblib': [],
        'pool': [],
        'processes': []
    }

    for size in test_sizes:
        print(f"\n{'=' * 40}")
        print(f"Test con {size} immagini")
        print(f"{'=' * 40}")

        base_seed = 42

        # Test sequenziale
        print("1. Test sequenziale...")
        random.seed(base_seed)
        start = time.time()
        seq_dir = compose_images_sequential(foreground, backgrounds, size, save_images)
        seq_time = time.time() - start
        results['sequential'].append(seq_time)
        print(f"   Sequenziale: {seq_time:.2f}s ({size / seq_time:.1f} img/s)")

        # Test joblib
        print("2. Test joblib...")
        random.seed(base_seed)
        start = time.time()
        joblib_dir = parallel_joblib(foreground, backgrounds, size, max_processes, save_images)
        joblib_time = time.time() - start
        results['joblib'].append(joblib_time)
        speedup_joblib = seq_time / joblib_time
        print(f"   Joblib: {joblib_time:.2f}s ({size / joblib_time:.1f} img/s) - Speedup: {speedup_joblib:.2f}x")

        # Test pool
        print("3. Test pool...")
        random.seed(base_seed)
        start = time.time()
        pool_dir = parallel_pool(foreground, backgrounds, size, max_processes, save_images)
        pool_time = time.time() - start
        results['pool'].append(pool_time)
        speedup_pool = seq_time / pool_time
        print(f"   Pool: {pool_time:.2f}s ({size / pool_time:.1f} img/s) - Speedup: {speedup_pool:.2f}x")

        # Test processes
        print("4. Test processes...")
        random.seed(base_seed)
        start = time.time()
        proc_dir = parallel_processes(foreground, backgrounds, size, max_processes, save_images)
        proc_time = time.time() - start
        results['processes'].append(proc_time)
        speedup_proc = seq_time / proc_time
        print(f"   Processes: {proc_time:.2f}s ({size / proc_time:.1f} img/s) - Speedup: {speedup_proc:.2f}x")

    csv_filename = save_results_to_csv(results, max_processes)

    return results, max_processes, csv_filename


def main():
    print("BENCHMARK IMAGE COMPOSITION")
    print("Confronto tra versione sequenziale e parallela")
    print(f"Core CPU disponibili: {multiprocessing.cpu_count()}")

    SAVE_IMAGES = False

    results, num_processes, csv_filename = run_benchmark(save_images=SAVE_IMAGES)

    if results is None:
        print("Benchmark fallito!")
        return

    create_plots(results, num_processes)
    print_summary(results, num_processes)

    print(f"\n{'=' * 60}")
    print("FILE GENERATI:")
    print(f"{'=' * 60}")
    print(f"Grafici: benchmark_results.png")
    print(f"Risultati: {csv_filename}")


if __name__ == "__main__":
    main()