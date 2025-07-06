cmake_minimum_required(VERSION 3.29)
project(kmeans)

set(CMAKE_CXX_STANDARD 20)

# Supporto OpenMP per macOS con percorso specifico
if(APPLE)
    set(LIBOMP_PATH "/opt/homebrew/opt/libomp")

    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PATH}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PATH}/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PATH}/lib/libomp.dylib")

    # Aggiungi direttamente i flag di compilazione
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

    # Aggiungi il percorso di link
    link_directories("${LIBOMP_PATH}/lib")
endif()

# Cerca OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP trovato e abilitato")
    add_definitions(-DUSE_OPENMP)
else()
    message(WARNING "OpenMP non trovato: l'implementazione parallela userà un fallback sequenziale")
endif()

# Un singolo eseguibile
add_executable(kmeans main.cpp
        main.cpp
        main.cpp
        main.cpp
        main.cpp)

# Assicurati di collegare la libreria OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans PUBLIC OpenMP::OpenMP_CXX)
else()
    # Fallback: collega direttamente la libreria
    target_link_libraries(kmeans PUBLIC "-L${LIBOMP_PATH}/lib -lomp")
endif()

# Ottimizzazione per il release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")