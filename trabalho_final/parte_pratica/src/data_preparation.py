import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import matplotlib.pyplot as plt

class DataPreparation:
    def __init__(self,
                 download_url: str = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
                 download_dir: str = "data",
                 work_dir: str = "data/physionet",
                 metadata_ecg_file: str = "ptbxl_database.csv",
                 images_dir: str = "ecg/images",
                 train_subdir: str = "train",
                 test_subdir: str = "test"):
        
        """Inicializa o construtor com os parâmetos

        Args:
            download_url (_type_, optional): url de download. Defaults to "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip".
            download_dir (str, optional): diretório de download. Defaults to "data/physionet".
            metadata_ecg_file (str, optional): arquivo de metadados e rótulos dos sinais de ecg. Defaults to "ptbxl_database.csv".
            images_dir (str, optional): diretório de imagens de ecg. Defaults to "ecg/images".
            train_subdirdir (str, optional): subdiretório de imagens de ecg de teste. Defaults to "train".
            test_subdirdir (str, optional): subdiretório de imagens de ecg de treinamento. Defaults to "test".
        """

        self.download_url = download_url
        self.download_dir = download_dir
        self.work_dir = work_dir
        self.metadata_ecg_file = os.path.join(work_dir, metadata_ecg_file)
        self.images_dir_train = os.path.join(images_dir, train_subdir)
        self.images_dir_test = os.path.join(images_dir, test_subdir)

    def download_data(self):
        """Baixa o arquivo zip do PhysioNet e extrai os arquivos."""

        print("Realizando download do dataset...")

        response = requests.get(self.download_url)
        if response.status_code == 200:
            print("Download concluído. Extraindo arquivos...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.download_dir)
            print("Extração completa.")
            # rename dir
            zip_dir = os.path.join(self.download_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
            os.rename(zip_dir, self.work_dir)
        else:
            raise Exception(f"Erro ao baixar a base de dados ecg da physionet: {response.status_code}")
        
    def load_metadata(self):
        """Carrega o arquivo de metadados."""

        print(f"Carregando metadados em {self.metadata_ecg_file} ...") 
        df_temp = pd.read_csv(self.metadata_ecg_file)
        print("Selecionando colunas relevantes ...")
        df_metadata = df_temp[["ecg_id", "scp_codes", "filename_hr"]]
        print(f"Metadados carregados. {len(df_metadata)} registros encontrados.")
        # Embaralhar
        df_shuffled = df_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        # Separar
        self.metadata_train = df_shuffled.iloc[:15000]
        print(f"Metadados de treinamento carregados. {len(self.metadata_train)} registros encontrados.")
        self.metadata_test = df_shuffled.iloc[15001:]
        print(f"Metadados de teste carregados. {len(self.metadata_test)} registros encontrados.")
    
    def _plot_ecg(self, ecg_signal: np.ndarray, save_path: str, dpi: int = None):
        """Plota o sinal de ECG e salva a imagem.

        Args:
            ecg_signal (np.ndarray): sinal de ECG a ser plotado.
            save_path (str): diretório para salvar a imagem.
            dpi (int): nitidez da imagem.
        """

        plt.figure(figsize=(10, 2))
        plt.plot(ecg_signal, color='black', linewidth=1)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

    def signals_to_images(self):
        """ Converte os sinais em imagens de ECG
        """
        
        signals_path = self.work_dir
        
        count_train = 0

        # treinamento
        for idx, row in self.metadata_train.iterrows():

            file_name = row['filename_hr']  # caminho relativo
            signal_file = os.path.join(signals_path, f"{file_name}.hea").replace(".hea", ".npy")

            if os.path.exists(signal_file):
                signal = np.load(signal_file)  # carrega o .npy
                save_path = os.path.join(self.images_dir_train, f"{row['ecg_id']}.png")
                self._plot_ecg(signal[0], save_path)  # usa o primeiro canal
                count_train += 1
            else:
                print(f"Arquivo não encontrado: {signal_file}")

        print(f"{count_train} imagens de ECG de treinamento geradas e salvas em {self.images_dir_train}.")

        count_test = 0

        # teste
        for idx, row in self.metadata_test.iterrows():

            file_name = row['filename_hr']  # caminho relativo
            signal_file = os.path.join(signals_path, f"{file_name}.hea").replace(".hea", ".npy")

            if os.path.exists(signal_file):
                signal = np.load(signal_file)  # carrega o .npy
                save_path = os.path.join(self.images_dir_test, f"{row['ecg_id']}.png")
                self._plot_ecg(signal[0], save_path)  # usa o primeiro canal
                count_test += 1
            else:
                print(f"Arquivo não encontrado: {signal_file}")

        print(f"{count_test} imagens de ECG de teste geradas e salvas em {self.images_dir_test}.")