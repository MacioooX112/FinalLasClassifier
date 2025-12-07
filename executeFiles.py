import subprocess
import sys

def run_command(command):
    """
    Uruchamia daną komendę w systemie.
    """
    try:
        print(f"Uruchamianie komendy: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Zakończono pomyślnie: {command}")
        print(f"Wyjście: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas wykonywania komendy: {command}")
        print(f"Wyjście: {e.stdout}")
        print(f"Błąd: {e.stderr}")
        
def main():
    # Lista komend do wykonania
    commands = [
        # "python las_generalize.py data.las percentage 10",
        "python las_flooring_divide.py general.las", #banaszak typ gówna
        #"python clustring"
        "python las_color_cluster2.py Objects/Remaining_points.las",
        "python point_cloud_classifier.py", #do przepuszczenia przez siec wszytskie dane i eksportujemy oznaczone
        "python Combine.py", #sklejamy wszystko
        "", #zwracamy plik, czyścimy pamięć i koniec
    ]
    
    for command in commands:
        run_command(command)
        
if __name__ == "__main__":
    main()
