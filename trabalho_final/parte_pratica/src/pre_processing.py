import os
import pandas as pd


class PreProcessing():
    def __init__(self,
                 work_dir: str = os.path.join("data", "physionet"),
                 images_dir: str = os.path.join("ecg", "images"),
                 csv_dir: str = os.path.join("ecg", "csv"),
                 train_subdir: str = "train",
                 test_subdir: str = "test"):
        
        self.work_dir = work_dir
        self.metadata_dir_train = os.path.join(csv_dir, train_subdir)
        self.metadata_dir_test = os.path.join(csv_dir, test_subdir)
        self.images_dir_train = os.path.join(images_dir, train_subdir)
        self.images_dir_test = os.path.join(images_dir, test_subdir)

    def _generate_path_images(self, path_images: str) -> list:
        """ Retorna uma lista com os ids das imagens sem a extensão ".png".
        """

        # Lista as imagens da pasta
        list_id_images = os.listdir(path_images)
        #print(list_images)

        # Filtra os arquivos .png e extrai os números (sem extensão)
        list_path_images = [os.path.join(path_images, os.path.splitext(f)[0]) for f in list_id_images if f.endswith(".png")]

        #print(list_path_images)
        return list_path_images
    
    def _create_dict_paths_images(self, list_paths_images: list) -> dict:
        """Cria um dicionário "id da imagem" (corresponde ao ecg_id) -> "path da imagem"
        """
        os_name = str(os.name) 
        # Detect OS
        if os_name == "nt":
            delimiter = '\\'
        else:
            delimiter = '/'

        dict_paths_images = {
            path_image.split(delimiter)[-1]: path_image
            for path_image in list_paths_images
        }

        return dict_paths_images
    
    def metadata_processing(self):
        """ Lê os datasets de teste e treinamento.
        """ 

        ################# Treinamento ##################
        self.df_metadata_train = pd.read_csv(os.path.join(self.metadata_dir_train, "df_metadata_train.csv"), sep=";")

        # Remove a coluna "Unnamed: 0" (lixo que foi ficando das transformações)
        self.df_metadata_train = self.df_metadata_train.drop(columns=["Unnamed: 0"])

        #print(self.df_metadata_train.columns)
        #print(self.df_metadata_train.sample(5))

        # Gerando os paths das imagens na pasta de treinamento
        list_paths_images_train = self._generate_path_images(self.images_dir_train)
        #print(list_paths_images_train)

        # Reorganiza os caminhos das imagens conforme a ordem da coluna "ecg_id"
        dict_paths_images_train_organized = self._create_dict_paths_images(list_paths_images_train)
        #print(dict_paths_images_train_organized)
        #print(len(dict_paths_images_train_organized))
        list_paths_images_train_organized = self.df_metadata_train["ecg_id"].apply(
            lambda x: dict_paths_images_train_organized.get(str(x))
        ).tolist()
        #print(list_paths_images_train_organized)

        # Adicionando coluna com os paths das imagens compatíveis com a colunas ecg_id
        self.df_metadata_train["paths_images_train"] = list_paths_images_train_organized
        
        ## Nova ordem das colunas
        new_order_columns_train = ["ecg_id", "filename_hr", "paths_images_train", "diagnosis"]

        # Reordenando colunas
        self.df_metadata_train = self.df_metadata_train[new_order_columns_train]

        # Salvar sobrescrevendo o dataset antigo de treinamento
        self.df_metadata_train.to_csv(os.path.join(self.metadata_dir_train, "df_metadata_train.csv"), sep=";")

        # Dataset final de treinamento com colunas organizadas
        print(self.df_metadata_train.sample(10))


        ################# Testes ##################
        self.df_metadata_test = pd.read_csv(os.path.join(self.metadata_dir_test, "df_metadata_test.csv"), sep=";")

        # Remove a coluna "Unnamed: 0" (lixo que foi ficando das transformações)
        self.df_metadata_test = self.df_metadata_test.drop(columns=["Unnamed: 0"])

        #print(self.df_metadata_test.columns)
        #print(self.df_metadata_test.sample(5))

        # Gerando os paths das imagens na pasta de testes
        list_paths_images_test = self._generate_path_images(self.images_dir_test)

        # Reorganiza os caminhos das imagens conforme a ordem da coluna "ecg_id"
        dict_paths_images_test_organized = self._create_dict_paths_images(list_paths_images_test)
        #print(dict_paths_images_test_organized)
        #print(len(dict_paths_images_test_organized))
        list_paths_images_test_organized = self.df_metadata_test["ecg_id"].apply(
            lambda x: dict_paths_images_test_organized.get(str(x))
        ).tolist()
        #print(list_paths_images_test_organized)

        # Adicionando coluna com os paths das imagens compatíveis com a colunas ecg_id
        self.df_metadata_test["paths_images_test"] = list_paths_images_test_organized
        
        ## Nova ordem das colunas
        new_order_columns_test = ["ecg_id", "filename_hr", "paths_images_test", "diagnosis"]

        # Reordenando colunas
        self.df_metadata_test = self.df_metadata_test[new_order_columns_test]

        # Salvar sobrescrevendo o dataset antigo de testes
        self.df_metadata_test.to_csv(os.path.join(self.metadata_dir_test, "df_metadata_test.csv"), sep=";")

        # Dataset final de testes com colunas organizadas
        print(self.df_metadata_test.sample(10))