from typing import Union, List # Per specificare campi multipli nell'input funzione
from typing import Literal # Per specificare campi multipli nell'input funzione
import os 
import numpy as np
from PIL import Image
import hashlib # per gli hash

def immagini_uguali(path_image1 : str,
                    path_image2: str,
                    verbose : bool = False, 
                   ) -> bool:
   """
   Questa funzione controlla che due immagini siano uguali.
   Prende in input il path delle due immagini, le importa e tramite numpy controlla che siano uguali pixel by pixel.

   Output:
      Restituisce True se uguali False se diverse.
   """   

   if not isinstance(verbose,bool):
      raise TypeError("Verbose deve essere un booleano")

   if not isinstance(path_image1,str):
      raise TypeError("path_image1 deve essere una stringa")

   if not isinstance(path_image2,str):
      raise TypeError("path_image2 deve essere una stringa")



   # Importare le immagini
   image1 = Image.open(path_image1)
   image2 = Image.open(path_image2)

   # Converte le immagini in array NumPy
   image1_array = np.array(image1)
   image2_array = np.array(image2)

   # Controlla se le due immagini sono uguali
   are_images_equal = np.array_equal(image1_array, image2_array)

   if verbose:
      # Stampa le dimensioni delle immagini
      print(f"Dimensione immagine 1: {image1_array.shape}")
      print(f"Dimensione immagine 2: {image2_array.shape}")

      if are_images_equal:
         print("Le immagini sono uguali.")
      else:
         print("Le immagini sono diverse.")

   return are_images_equal

def immagini_simili(path_image1 : str,
                    path_image2: str,
                    tolleranza : float =  1e-5,
                    verbose : bool = False, 
                   ) -> bool:
   """
   Questa funzione controlla che due immagini siano uguali.
   Prende in input il path delle due immagini, le importa e tramite numpy controlla che siano uguali pixel by pixel.


   Output:
      Restituisce True se uguali False se diverse.
   """   

   if not isinstance(verbose,bool):
      raise TypeError("Verbose deve essere un booleano")

   if not isinstance(path_image1,str):
      raise TypeError("path_image1 deve essere una stringa")

   if not isinstance(path_image2,str):
      raise TypeError("path_image2 deve essere una stringa")

   if not isinstance(tolleranza,float):
      raise TypeError("tolleranza deve essere un float")

   

   # Importare le immagini
   image1 = Image.open(path_image1)
   image2 = Image.open(path_image2)

   # Converte le immagini in array NumPy
   image1_array = np.array(image1)
   image2_array = np.array(image2)


   # Ridimensiona l'immagine 2 alle dimensioni di image1
   image2_resized = image2.resize(image1.size)

   # Converte l'immagine ridimensionata in array NumPy
   image2_resized_array = np.array(image2_resized)

   # Confronta le immagini con tolleranza
   are_images_equal = np.allclose(image1_array, image2_resized_array, atol=tolleranza)

   if verbose:
      # Stampa le dimensioni delle immagini
      print(f"Dimensione immagine 1: {image1_array.shape}")
      print(f"Dimensione immagine 2: {image2_array.shape}")

      if are_images_equal:
         print("Le immagini sono simili.")
      else:
         print("Le immagini sono diverse.")

   return are_images_equal

def immagini_uguali_in_cartella( path_directory: str,
                                 tipo_controllo: Literal["duplicati", "simili"],
                                 tolleranza : float =  1e-5,
                               ):
    """
    Confronta tutte le immagini in una cartella e trova immagini duplicate o simili.

    :param path_directory: Percorso della cartella contenente le immagini.
    :param tipo_controllo: "duplicati" per confronto esatto, "simili" per confronto con tolleranza.
    :return: Lista di coppie di immagini uguali o simili.
    """
    nomi_file = []
    path_file = []
    # Legge tutte le immagini nella cartella
    for filename in os.listdir(path_directory):
        file_path = os.path.join(path_directory, filename)
        try:
            nomi_file.append(filename)
            path_file.append(file_path)
        except Exception as e:
            print(f"Errore nell'aprire {file_path}: {e}")

    immagini_duplicate = []

    # Confronta ogni coppia di immagini
    for i in range(len(nomi_file)):
        for j in range(i + 1, len(nomi_file)):
            img1,nome_img1 = path_file[i],nomi_file[i]
            img2,nome_img2 = path_file[j],nomi_file[j]

            if tipo_controllo == "duplicati":
                # Controllo esatto pixel per pixel
                if immagini_uguali(img1,img2):
                    immagini_duplicate.append((nome_img1,nome_img2))

            elif tipo_controllo == "simili":
                # Controllo con una tolleranza di differenza tra pixel
                if immagini_simili(img1,img2,tolleranza=tolleranza):
                    immagini_duplicate.append((nome_img1,nome_img2))

    return immagini_duplicate

def get_image_hash(image_array):
    """Calcola un hash per l'immagine."""
    image_bytes = image_array.tobytes()  # Converte l'immagine in una sequenza di byte
    return hashlib.md5(image_bytes).hexdigest()  # Genera l'hash MD5

def immagini_uguali_tensorflow_data(dataset1,dataset2):
   # Ottieni gli hash delle immagini nel train set con i nomi dei file
   train_hashes = {}
   for i in range(len(dataset1)):
      images, _ = dataset1[i]  # Ignoriamo le etichette
      filenames = dataset1.filenames[i * dataset1.batch_size : (i + 1) * dataset1.batch_size]
      
      for img, filename in zip(images, filenames):
         img_hash = get_image_hash((img * 255).astype(np.uint8))  # Converti i valori di pixel a [0, 255]
         if img_hash in train_hashes:
               train_hashes[img_hash].append(filename)  # Aggiungi il file duplicato alla lista
         else:
               train_hashes[img_hash] = [filename]  # Crea una nuova lista con il primo file

   # Controlla le immagini nel test set
   duplicate_files = []
   for i in range(len(dataset2)):
      images, _ = dataset2[i]
      filenames = dataset2.filenames[i * dataset2.batch_size : (i + 1) * dataset2.batch_size]

      for img, filename in zip(images, filenames):
         img_hash = get_image_hash((img * 255).astype(np.uint8))
         if img_hash in train_hashes:
               for train_file in train_hashes[img_hash]:  # Controlla tutti i duplicati
                  duplicate_files.append((filename, train_file))  # Salva ogni coppia duplicata

   # Stampa i risultati
   if duplicate_files:
      print(f"Attenzione! Sono state trovate {len(duplicate_files)} immagini duplicate tra train e test.")
      for test_file, train_file in duplicate_files:
         print(f"Dataset 1: {test_file}  <->  Dataset 2: {train_file}")
   else:
      print("Nessuna immagine duplicata tra train e test.")
