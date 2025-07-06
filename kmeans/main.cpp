/**
 * K-means clustering - sequenziale e parallelo
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <ctime>

// Includi OpenMP se disponibile
#ifdef USE_OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x)
#endif

using namespace std;

/**
 * Punto 2D
 */
class Point {
public:
    double x, y;
    int cluster;

public:
    Point(double x, double y) {
        this->x = x;
        this->y = y;
        this->cluster = -1;
    }

    // Distanza euclidea al quadrato
    double distanceSquared(const Point& centroid) const {
        double dx = x - centroid.x;
        double dy = y - centroid.y;
        return dx * dx + dy * dy;
    }
};

/**
 * Salva risultati in CSV
 */
void writeToCSV(const vector<Point>& points, const vector<Point>& centroids, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Errore nell'aprire il file: " << filename << endl;
        return;
    }

    file << "type,x,y,cluster\n";

    for (const auto& point : points) {
        file << "point," << point.x << "," << point.y << "," << point.cluster << "\n";
    }

    for (size_t i = 0; i < centroids.size(); i++) {
        file << "centroid," << centroids[i].x << "," << centroids[i].y << "," << i << "\n";
    }

    file.close();
    cout << "Risultati salvati in " << filename << endl;
}

/**
 * Salva performance
 */
void savePerformance(const string& filename, const string& version, int numPoints,
                    int k, int threads, double timeMs, double sse) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) return;

    file.seekp(0, ios::end);
    if (file.tellp() == 0) {
        file << "version,num_points,k,threads,time_ms,sse\n";
    }

    file << version << "," << numPoints << "," << k << "," << threads << ","
         << timeMs << "," << sse << "\n";
    file.close();
}

/**
 * Calcola SSE
 */
double calculateSSE(const vector<Point>& points, const vector<Point>& centroids) {
    double sse = 0.0;
    for (const auto& point : points) {
        sse += point.distanceSquared(centroids[point.cluster]);
    }
    return sse;
}

/**
 * K-means sequenziale
 */
void kmeansSequential(vector<Point>& points, vector<Point>& centroids, int k) {
    bool changed = true;
    int iteration = 0;

    while (changed) {
        changed = false;

        // Assegna punti ai cluster
        for (auto& point : points) {
            double minDist = numeric_limits<double>::max();
            int bestCluster = -1;

            for (int j = 0; j < k; j++) {
                double dist = point.distanceSquared(centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            if (point.cluster != bestCluster) {
                point.cluster = bestCluster;
                changed = true;
            }
        }

        // Aggiorna centroidi
        if (changed) {
            vector<double> sumX(k, 0.0), sumY(k, 0.0);
            vector<int> count(k, 0);

            for (const auto& point : points) {
                sumX[point.cluster] += point.x;
                sumY[point.cluster] += point.y;
                count[point.cluster]++;
            }

            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    centroids[j].x = sumX[j] / count[j];
                    centroids[j].y = sumY[j] / count[j];
                }
            }
        }

        iteration++;
    }

    cout << "K-means sequenziale completato dopo " << iteration << " iterazioni" << endl;
}

/**
 * K-means parallelo
 */
void kmeansParallel(vector<Point>& points, vector<Point>& centroids, int k, int numThreads) {
    bool changed = true;
    int iteration = 0;

#ifdef USE_OPENMP
    omp_set_num_threads(numThreads);
#endif

    while (changed) {
        changed = false;

        vector<double> sumX(k, 0.0), sumY(k, 0.0);
        vector<int> count(k, 0);

        // Assegna punti ai cluster (parallelo)
#ifdef USE_OPENMP
        #pragma omp parallel
        {
            vector<double> localSumX(k, 0.0), localSumY(k, 0.0);
            vector<int> localCount(k, 0);
            bool localChanged = false;

            #pragma omp for
            for (size_t i = 0; i < points.size(); i++) {
                double minDist = numeric_limits<double>::max();
                int bestCluster = -1;

                for (int j = 0; j < k; j++) {
                    double dist = points[i].distanceSquared(centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }

                if (points[i].cluster != bestCluster) {
                    points[i].cluster = bestCluster;
                    localChanged = true;
                }

                localSumX[bestCluster] += points[i].x;
                localSumY[bestCluster] += points[i].y;
                localCount[bestCluster]++;
            }

            #pragma omp critical
            {
                if (localChanged) changed = true;
                for (int j = 0; j < k; j++) {
                    sumX[j] += localSumX[j];
                    sumY[j] += localSumY[j];
                    count[j] += localCount[j];
                }
            }
        }
#else
        // Versione sequenziale se OpenMP non c'è
        for (auto& point : points) {
            double minDist = numeric_limits<double>::max();
            int bestCluster = -1;

            for (int j = 0; j < k; j++) {
                double dist = point.distanceSquared(centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            if (point.cluster != bestCluster) {
                point.cluster = bestCluster;
                changed = true;
            }

            sumX[bestCluster] += point.x;
            sumY[bestCluster] += point.y;
            count[bestCluster]++;
        }
#endif

        // Aggiorna centroidi
        if (changed) {
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    centroids[j].x = sumX[j] / count[j];
                    centroids[j].y = sumY[j] / count[j];
                }
            }
        }

        iteration++;
    }

    cout << "K-means parallelo completato dopo " << iteration << " iterazioni" << endl;
}

/**
 * Genera punti casuali
 */
void generateDataset(vector<Point>& points, int numPoints, int seed) {
    srand(seed);
    points.clear();
    points.reserve(numPoints);

    for (int i = 0; i < numPoints; i++) {
        points.emplace_back(rand() % 1000, rand() % 1000);
    }
}

/**
 * Inizializza centroidi
 */
void initializeCentroids(vector<Point>& centroids, int k, int seed) {
    srand(seed + 100);
    centroids.clear();
    centroids.reserve(k);

    for (int i = 0; i < k; i++) {
        centroids.emplace_back(rand() % 1000, rand() % 1000);
    }
}

/**
 * Copia i punti
 */
vector<Point> copyPoints(const vector<Point>& original) {
    vector<Point> copy;
    copy.reserve(original.size());
    for (const auto& point : original) {
        copy.emplace_back(point.x, point.y);
    }
    return copy;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Uso: " << argv[0] << " <num_punti> <num_cluster> [num_thread]" << endl;
        return 1;
    }

    int numPoints = atoi(argv[1]);
    int k = atoi(argv[2]);
    int numThreads = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();

    cout << "=== CONFRONTO K-MEANS: SEQUENZIALE VS PARALLELO ===" << endl;
    cout << "Punti: " << numPoints << ", Cluster: " << k << ", Thread: " << numThreads << endl;
    cout << "=" << string(50, '=') << endl;

    // Genera dataset
    vector<Point> originalPoints;
    generateDataset(originalPoints, numPoints, numPoints + k);
    cout << "Dataset generato: " << numPoints << " punti" << endl;

    // ===== TEST SEQUENZIALE =====
    cout << "\n--- ESECUZIONE SEQUENZIALE ---" << endl;

    vector<Point> pointsSeq = copyPoints(originalPoints);
    vector<Point> centroidsSeq;
    initializeCentroids(centroidsSeq, k, numPoints + k);

    auto startSeq = chrono::high_resolution_clock::now();
    kmeansSequential(pointsSeq, centroidsSeq, k);
    auto endSeq = chrono::high_resolution_clock::now();

    double timeSeq = chrono::duration<double, milli>(endSeq - startSeq).count();
    double sseSeq = calculateSSE(pointsSeq, centroidsSeq);

    cout << "Tempo sequenziale: " << timeSeq << " ms" << endl;
    cout << "SSE sequenziale: " << sseSeq << endl;

    // ===== TEST PARALLELO =====
    cout << "\n--- ESECUZIONE PARALLELA ---" << endl;

    vector<Point> pointsPar = copyPoints(originalPoints);
    vector<Point> centroidsPar;
    initializeCentroids(centroidsPar, k, numPoints + k);

    auto startPar = chrono::high_resolution_clock::now();
    kmeansParallel(pointsPar, centroidsPar, k, numThreads);
    auto endPar = chrono::high_resolution_clock::now();

    double timePar = chrono::duration<double, milli>(endPar - startPar).count();
    double ssePar = calculateSSE(pointsPar, centroidsPar);

    cout << "Tempo parallelo: " << timePar << " ms" << endl;
    cout << "SSE parallelo: " << ssePar << endl;

    // ===== ANALISI PERFORMANCE =====
    cout << "\n--- ANALISI PERFORMANCE ---" << endl;
    double speedup = timeSeq / timePar;
    double efficiency = speedup / numThreads;

    cout << "Speedup: " << speedup << "x" << endl;
    cout << "Efficienza: " << efficiency << endl;

    // ===== SALVA RISULTATI =====
    writeToCSV(pointsSeq, centroidsSeq, "kmeans_sequential.csv");
    writeToCSV(pointsPar, centroidsPar, "kmeans_parallel.csv");

    savePerformance("kmeans_performance.csv", "sequential", numPoints, k, 1, timeSeq, sseSeq);
    savePerformance("kmeans_performance.csv", "parallel", numPoints, k, numThreads, timePar, ssePar);

    cout << "\n=== COMPLETATO ===" << endl;
    cout << "File generati:" << endl;
    cout << "- kmeans_sequential.csv (risultati sequenziali)" << endl;
    cout << "- kmeans_parallel.csv (risultati paralleli)" << endl;
    cout << "- kmeans_performance.csv (performance)" << endl;

    return 0;
}