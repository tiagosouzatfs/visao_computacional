import os
import pandas as pd
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import json
import wfdb

class DataPreparation:
    def __init__(self,
                 download_url: str = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
                 download_dir: str = "data",
                 work_dir: str = os.path.join("data", "physionet"),
                 metadata_ecg_file: str = "ptbxl_database.csv",
                 images_dir: str = os.path.join("ecg", "images"),
                 csv_dir: str = os.path.join("ecg", "csv"),
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
        self.metadata_dir_train = os.path.join(csv_dir, train_subdir)
        self.metadata_dir_test = os.path.join(csv_dir, test_subdir)
        self.images_dir_train = os.path.join(images_dir, train_subdir)
        self.images_dir_test = os.path.join(images_dir, test_subdir)

    def download_data(self):
        """Baixa o arquivo zip do PhysioNet e extrai os arquivos.
        """

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
        
    def load_metadata(self) -> list:
        """Carrega o arquivo de metadados.
        """

        print(f"Carregando metadados em {self.metadata_ecg_file} ...") 
        df_temp = pd.read_csv(self.metadata_ecg_file)
        print("Selecionando colunas relevantes ...")
        df_metadata = df_temp[["ecg_id", "scp_codes", "filename_hr"]]
        print(f"Metadados carregados. {len(df_metadata)} registros encontrados.")
        
        # Renomear a coluna de código de diagnóstico
        df_metadata = df_metadata.rename(columns={"scp_codes": "diagnosis"})
        
        """ Metadados encontrados em outra planilha csv: scp_statements.csv
        Alterar a coluna diagnosis para as 3 classes principais: normal, infarto e outros (fibrilações, arritmias, bloqueios e etc...)
        NORM => Normal ECG
        MI => Myocardial Infarction
            IMI	Myocardial Infarction	inferior myocardial infarction
            AMI	Myocardial Infarction	anteroseptal myocardial infarction
            IMI	Myocardial Infarction	inferolateral myocardial infarction
            AMI	Myocardial Infarction	anterior myocardial infarction
            AMI	Myocardial Infarction	anterolateral myocardial infarction
            LMI	Myocardial Infarction	lateral myocardial infarction
            IMI	Myocardial Infarction	inferoposterolateral myocardial infarction
            IMI	Myocardial Infarction	inferoposterior myocardial infarction
            PMI	Myocardial Infarction	posterior myocardial infarction
        # Other => ST/T Change, Conduction Disturbance, Hypertrophy e etc...
        """
        list_diagnosis = ["NORM", "IMI", "AMI", "LMI", "PMI", "Other"]

        # Trocar aspas simples por aspas duplas para o json.loads() funcionar corretamente
        df_metadata['diagnosis'] = df_metadata['diagnosis'].apply(lambda x: json.loads(x.replace("'", '"')))

        # Função para substituir o dicionário por uma das palavras de interesse
        def extract_diagnosis(coluna_df):
            for chave in coluna_df:
                if chave in list_diagnosis:
                    return chave
            return "Other"  #"Other" para preencher com um valor padrão

        # Aplicar ao DataFrame
        df_metadata['diagnosis'] = df_metadata['diagnosis'].apply(extract_diagnosis)
        #print(df_metadata.sample(10))

        list_infartations = ["IMI", "AMI", "LMI", "PMI"]
        df_metadata['diagnosis'] = df_metadata['diagnosis'].replace(list_infartations, 'MI')
        #print(df_metadata.sample(10))
        #print(df_metadata[df_metadata['diagnosis'] == "Other"])
        #print(df_metadata[df_metadata['diagnosis'] == "MI"])
        #print(df_metadata[df_metadata['diagnosis'].isin(['MI', 'NORM'])])
        #print(df_metadata[~df_metadata['diagnosis'].isin(['MI', 'NORM', 'Other'])])
        #print(len(df_metadata[df_metadata['diagnosis'].isin(['MI', 'NORM', 'Other'])]))

        # Embaralhar
        df_shuffled = df_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Separar
        self.df_metadata_train = df_shuffled.iloc[:15000]
        #print(self.metadata_train.head())
        print(f"Metadados de treinamento carregados. {len(self.df_metadata_train)} registros encontrados.")
        self.df_metadata_test = df_shuffled.iloc[15001:]
        #print(self.metadata_test.head())
        print(f"Metadados de teste carregados. {len(self.df_metadata_test)} registros encontrados.")

        # Salvando datasets
        print("Salvando dataset de treinamento ...")
        self.df_metadata_train.to_csv(os.path.join(self.metadata_dir_train, "df_metadata_train.csv"), sep=';', encoding='utf-8')
        print("Salvando dataset de teste ...")
        self.df_metadata_test.to_csv(os.path.join(self.metadata_dir_test, "df_metadata_test.csv"), sep=';', encoding='utf-8')

    # Usa apenas o primeiro canal
    def _ecg_to_png(self, record_path: str, output_path: str, channel: int=0): 
        """
        Lê um arquivo ECG .hea + .dat/.mat e salva como imagem PNG.
        
        Parâmetros:
            record_path (str): caminho base do arquivo (sem extensão).
            output_path (str): caminho do arquivo PNG de saída.
            channel (int): índice do canal a ser plotado (default = 0).
        """
        # Lê o registro
        record = wfdb.rdrecord(record_path)

        # Extrai o sinal (formato: array [amostras, canais])
        signal = record.p_signal

        # Extrai o canal desejado
        ecg = signal[:, channel]

        # Cria a imagem
        plt.figure(figsize=(10, 3))
        plt.plot(ecg, color='black', linewidth=0.8)
        plt.axis('off')  # remove os eixos
        plt.tight_layout()

        # Salva como imagem
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    def signals_to_images(self):
        """ Converte os sinais em imagens de ECG.
        """
        
        path_images_dir = [self.images_dir_train, self.images_dir_test]   
        list_metadata_dataset = [self.df_metadata_train, self.df_metadata_test]
        i=0

        for dataset in list_metadata_dataset:
            count_plot = 0

            for idx, row in dataset.iterrows():
                # Limite definido de entrada para o dataset de teste
                if count_plot == 15001:
                    break
                    
                file_name = row['filename_hr']  # caminho relativo
                signal_file = os.path.join(self.work_dir, file_name)
                signal_dir = os.path.dirname(signal_file)
                print(signal_dir)
                
                if os.path.exists(signal_dir):
                    save_path = os.path.join(path_images_dir[i], f"{row['ecg_id']}.png")
                    self._ecg_to_png(signal_file, save_path)
                    count_plot += 1
                else:
                    print(f"Arquivo não encontrado: {signal_file}")
            i += 1