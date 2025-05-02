import os
import pandas as pd
import autokeras as ak
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array


class AKModel():
    def __init__(self,
                 images_dir: str = os.path.join("ecg", "images"),
                 csv_dir: str = os.path.join("ecg", "csv"),
                 train_subdir: str = "train",
                 test_subdir: str = "test"):
        
        self.metadata_dir_train = os.path.join(csv_dir, train_subdir)
        self.metadata_dir_test = os.path.join(csv_dir, test_subdir)
        self.images_dir_train = os.path.join(images_dir, train_subdir)
        self.images_dir_test = os.path.join(images_dir, test_subdir)

    def run(self):

        df_metadata_train = pd.read_csv(os.path.join(self.metadata_dir_train, "df_metadata_train.csv"), sep=";")
        df_metadata_test = pd.read_csv(os.path.join(self.metadata_dir_test, "df_metadata_test.csv"), sep=";")

        df_metadata_train["paths_images_train"] = df_metadata_train["paths_images_train"].astype(str) + ".png"
        df_metadata_test["paths_images_test"] = df_metadata_test["paths_images_test"].astype(str) + ".png"

        print("Leitura de csvs")

        # Carregar imagens manualmente a partir dos caminhos
        #def load_images(paths, target_size=(224, 224)):
        def load_images(paths, target_size=(128, 128)):
            return np.array([
                img_to_array(load_img(path, target_size=target_size)) / 255.0
                for path in paths
            ])

        # Carrega os dados
        X_train = load_images(df_metadata_train["paths_images_train"])
        y_train = df_metadata_train["diagnosis"].values

        #X_test = load_images(df_metadata_test["paths_images_test"])
        #y_test = df_metadata_test["diagnosis"].values

        clf = ak.ImageClassifier(overwrite=True, max_trials=2)
        
        clf.fit(X_train, y_train, epochs=15)
        
        #loss, acc = clf.evaluate(X_test, y_test)
        #print(f"Acurácia no conjunto de teste: {acc:.4f}")
