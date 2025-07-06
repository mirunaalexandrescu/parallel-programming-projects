import subprocess
import time
import matplotlib.pyplot as plt
import re

def extract_speedup_from_output(output):
    """Estrae lo speedup dall'output del programma C++"""
    lines = output.split('\n')
    for line in lines:
        if "Speedup:" in line:
            match = re.search(r'Speedup:\s*([0-9.]+)x', line)
            if match:
                return float(match.group(1))
    return None

def test_configuration(points, clusters):
    """Testa una configurazione e estrae i veri speedup"""
    print("=== Testing {} points, {} clusters ===".format(points, clusters))
    
    results = {}
    
    # Thread counts da testare
    thread_counts = [1, 2, 4, 8, 16, 32]
    
    for threads in thread_counts:
        print("Testing {} threads...".format(threads))
        speedups = []
        
        for i in range(5):
            try:
                result = subprocess.run("./kmeans {} {} {}".format(points, clusters, threads), 
                                      shell=True, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    speedup = extract_speedup_from_output(result.stdout)
                    if speedup is not None:
                        speedups.append(speedup)
                        print("  Test {}: speedup {:.2f}x".format(i+1, speedup))
                    else:
                        print("  Test {}: could not extract speedup".format(i+1))
                else:
                    print("  Test {}: execution failed".format(i+1))
                    
            except subprocess.TimeoutExpired:
                print("  Test {}: timeout".format(i+1))
            except Exception as e:
                print("  Test {}: error - {}".format(i+1, e))
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            results[threads] = avg_speedup
            print("  Average speedup: {:.2f}x".format(avg_speedup))
        else:
            print("  No valid results for {} threads".format(threads))
    
    return results

def save_results(all_results, filename):
    """Salva risultati in formato CSV"""
    with open(filename, "w") as file:
        file.write("Points,Clusters,Threads,Speedup\n")
        
        for config, results in all_results.items():
            points, clusters = config
            for threads, speedup in results.items():
                file.write("{},{},{},{:.2f}\n".format(points, clusters, threads, speedup))

def create_speedup_plots(all_results):
    """Crea grafici di speedup"""
    
    # Configura il plot
    points_configs = list(set([config[0] for config in all_results.keys()]))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plot_idx = 0
    for points in sorted(points_configs):
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        
        # Trova tutte le configurazioni con questo numero di punti
        configs_for_points = [config for config in all_results.keys() if config[0] == points]
        
        for i, config in enumerate(sorted(configs_for_points)):
            points_val, clusters = config
            results = all_results[config]
            
            if results:  # Se ci sono risultati
                threads = sorted(results.keys())
                speedup_values = [results[t] for t in threads]
                
                ax.plot(threads, speedup_values, 'o-', color=colors[i], 
                       linewidth=2, markersize=6, label='{} clusters'.format(clusters))
        
        # Linea ideale
        max_threads = 32
        ax.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, linewidth=1, label='Ideal')
        
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup')
        ax.set_title('SpeedUp {}k points'.format(points//1000))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max_threads)
        ax.set_ylim(1, max(8, max_threads//4))
        
        # Scala logaritmica per l'asse x
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
        
        plot_idx += 1
    
    # Rimuovi subplot vuoti
    for i in range(plot_idx, 4):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('kmeans_speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Speedup plots saved to: kmeans_speedup_analysis.png")

def create_summary_plot(all_results):
    """Crea grafico riassuntivo"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Raggruppa per numero di punti
    points_groups = {}
    for config, results in all_results.items():
        points, clusters = config
        if points not in points_groups:
            points_groups[points] = {}
        points_groups[points][clusters] = results
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (points, cluster_results) in enumerate(sorted(points_groups.items())):
        clusters_list = []
        speedups_8t = []
        
        for clusters, results in sorted(cluster_results.items()):
            if 8 in results:  # Speedup con 8 thread
                clusters_list.append(clusters)
                speedups_8t.append(results[8])
        
        if clusters_list:
            ax.plot(clusters_list, speedups_8t, 'o-', color=colors[i], 
                   linewidth=2, markersize=8, label='{}k points'.format(points//1000))
    
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Speedup (8 threads)')
    ax.set_title('SpeedUp with 8 Threads')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_speedup_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Summary plot saved to: kmeans_speedup_summary.png")

def main():
    # Configurazioni da testare
    configurations = [
        (100000, 25),   # 100k punti, 25 cluster
        (100000, 50),   # 100k punti, 50 cluster  
        (100000, 100),  # 100k punti, 100 cluster
        (500000, 50),   # 500k punti, 50 cluster
        (500000, 100),  # 500k punti, 100 cluster
        (500000, 250),  # 500k punti, 250 cluster
    ]
    
    print("=== K-MEANS SPEEDUP ANALYSIS ===")
    print("Extracting real speedup from C++ output")
    print("Configurations:")
    for points, clusters in configurations:
        print("  - {} points, {} clusters".format(points, clusters))
    print("Threads: 1, 2, 4, 8, 16, 32")
    print("5 iterations per configuration")
    print("=" * 50)
    
    all_results = {}
    
    for points, clusters in configurations:
        results = test_configuration(points, clusters)
        if results:
            all_results[(points, clusters)] = results
            
            # Stampa riassunto
            print("Summary for {} points, {} clusters:".format(points, clusters))
            for threads in sorted(results.keys()):
                print("  {} threads: speedup {:.2f}x".format(threads, results[threads]))
            print("-" * 30)
        else:
            print("No results for {} points, {} clusters".format(points, clusters))
    
    if all_results:
        # Salva risultati
        save_results(all_results, "kmeans_speedup_data.csv")
        print("Results saved to: kmeans_speedup_data.csv")
        
        # Crea grafici
        print("Creating speedup plots...")
        create_speedup_plots(all_results)
        
        print("Creating summary plot...")
        create_summary_plot(all_results)
        
        print("\n=== ANALYSIS COMPLETED ===")
        
        # Trova migliori risultati
        best_speedup = 0
        best_config = None
        best_threads = 0
        
        for config, results in all_results.items():
            for threads, speedup in results.items():
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_config = config
                    best_threads = threads
        
        if best_config:
            points, clusters = best_config
            print("Best speedup: {:.2f}x with {} threads ({} points, {} clusters)".format(
                best_speedup, best_threads, points, clusters))
    else:
        print("No valid results obtained!")

if __name__ == "__main__":
    main()
