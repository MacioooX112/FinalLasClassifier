import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import time

def classify_terrain_kmeans(las_file_path, K_neighbors=15, num_clusters=3, output_file=None):
    """
    Sklasyfikuje punkty chmury (podłoża) na 3 klastry (np. trawa, ulica, szyny) 
    używając K-Means na bazie cech geometrycznych i atrybutowych.

    Cechy wykorzystane: Wysokość Z, Intensywność, Kolor (R, G, B), Logarytm Lokalnej Wariancji Z (gładkość).

    Args:
        las_file_path (str): Ścieżka do wejściowego pliku .las.
        K_neighbors (int): Liczba najbliższych sąsiadów do obliczenia lokalnej gładkości.
        num_clusters (int): Docelowa liczba klastrów (domyślnie 3).
        output_file (str, optional): Ścieżka do zapisu pliku wyjściowego.
    
    Returns:
        str: Ścieżka do zapisanego pliku.
    """
    start_time = time.time()
    
    print(f"Reading LiDAR data from: {las_file_path}")
    las = laspy.read(las_file_path)
    
    # Sprawdzenie, czy dane mają wymagane atrybuty
    required_dims = ['X', 'Y', 'Z', 'red', 'green', 'blue', 'intensity']
    for dim in required_dims:
        if dim not in las.point_format.dimension_names:
            print(f"BŁĄD: Wymiar '{dim}' nie został znaleziony w pliku LAS. Klasyfikacja jest niemożliwa.")
            return None

    # Ekstrakcja danych
    X, Y, Z = np.array(las.x), np.array(las.y), np.array(las.z)
    R, G, B = np.array(las.red), np.array(las.green), np.array(las.blue)
    Intensity = np.array(las.intensity)
    
    all_coords = np.vstack((X, Y, Z)).T
    N = len(X)
    print(f"  Total points: {N:,}")
    print(f"  Calculating local features with K={K_neighbors}...")
    
    # ----------------------------------------------------
    # KROK 1: EKSTRAKCJA CECH LOKALNYCH (Gładkość/Chropowatość)
    # ----------------------------------------------------
    
    # Znajdź K najbliższych sąsiadów dla każdego punktu
    # Używamy tylko X i Y do obliczenia odległości w płaszczyźnie
    nbrs = NearestNeighbors(n_neighbors=K_neighbors, algorithm='auto').fit(all_coords[:, :2])
    # Odległości (nieużywane) i indeksy sąsiadów
    distances, indices = nbrs.kneighbors(all_coords[:, :2])

    local_variance = np.zeros(N)

    # Obliczenie wariancji wysokości (Z) w otoczeniu każdego punktu
    for i in range(N):
        neighbor_indices = indices[i, :]
        neighbor_Zs = Z[neighbor_indices]
        
        # Wariancja jest miarą lokalnej chropowatości/gładkości
        local_variance[i] = np.var(neighbor_Zs)

    # Logarytm wariancji poprawia rozkład danych do klasteryzacji
    epsilon = 1e-6 
    log_variance = np.log(local_variance + epsilon)
    
    print("  Local features calculated.")

    # ----------------------------------------------------
    # KROK 2: PRZYGOTOWANIE I SKALOWANIE DANYCH
    # ----------------------------------------------------
    
    # Macierz cech: Z, Intensywność, R, G, B, Logarytm Wariancji
    feature_matrix = np.column_stack([
        Z, 
        Intensity, 
        R, 
        G, 
        B, 
        log_variance
    ])

    # Skalowanie cech jest KLUCZOWE dla K-Means
    # Zapewnia, że żadna cecha (np. R, G, B z zakresu 0-65535) nie dominuje nad innymi (np. wariancja)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    print("  Features scaled.")
    
    # ----------------------------------------------------
    # KROK 3: KLASTERyZACJA K-MEANS
    # ----------------------------------------------------
    
    print(f"  Running K-Means clustering with K={num_clusters}...")
    
    # n_init=10 jest zalecane, aby znaleźć najlepsze klastry.
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, verbose=0)
    
    # Przypisanie etykiet klastrów do punktów
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # ----------------------------------------------------
    # KROK 4: ZAPIS WYNIKÓW
    # ----------------------------------------------------
    
    # Przypisanie numerów klastrów (0, 1, 2) do atrybutu 'classification'
    # Zgodnie ze standardem LAS Classification wymaga typu np.uint8
    las.classification = cluster_labels.astype(np.uint8)

    # Ustalenie nazwy pliku wyjściowego
    if output_file is None:
        base_name = os.path.splitext(las_file_path)[0]
        output_file = f"{base_name}_classified_kmeans.las"
        
    # Zapis
    las.write(output_file)
    
    end_time = time.time()
    
    # Podsumowanie
    print("-" * 50)
    print(" Klasyfikacja zakończona pomyślnie!")
    print(f"  Całkowity czas: {end_time - start_time:.2f} sekund")
    print(f"  Liczba punktów w klastrach:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Klaster {u}: {c:,} punktów")
    print(f"  Sklasyfikowany plik zapisano jako: {output_file}")
    print("-" * 50)
    
    # Wskazówka dla użytkownika (interpretacja klastrów)
    print("\n UWAGA: Wartości 0, 1, 2 to automatycznie wygenerowane klastry. Musisz je zinterpretować ręcznie w wizualizatorze (np. CloudCompare) i przypisać do klas: Trawa, Ulica, Szyny.")
    
    return output_file

# --- PRZYKŁAD UŻYCIA ---
if __name__ == '__main__':
    # Zmień ścieżkę do swojego pliku .las
    input_file = r"C:\Users\MaciooX\OneDrive\Pulpit\Hackaton\podlogi\Objects\Class2_points.las"  
    
    if os.path.exists(input_file):
        classified_file = classify_terrain_kmeans(
            las_file_path=input_file,
            K_neighbors=20  # Lepsza wartość dla stabilności obliczeń wariancji
        )
        print(f"Plik gotowy do wizualizacji: {classified_file}")
    else:
        print(f"Błąd: Nie znaleziono pliku {input_file}. Upewnij się, że plik istnieje.")