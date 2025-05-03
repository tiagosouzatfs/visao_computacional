from data_preparation import DataPreparation
from pre_processing import PreProcessing
from ak_model import AKModel

def main():

    # Cria a pipeline de ETL
    pipeline_etl = DataPreparation()

    # 1. Baixa os dados do dataset e seus metadados
    pipeline_etl.download_data()

    # 2. Carrega os metadados principais
    pipeline_etl.load_metadata()

    # 3. Gera as imagens dos sinais para o treinamento e teste
    pipeline_etl.signals_to_images()


    # Cria a pipeline de pré-processamento
    pipeline_pre_processing = PreProcessing()

    # 4. Ajusta o metadados de processamento para serem utilizados no modelo
    pipeline_pre_processing.metadata_processing()


    # Cria a pipeline de execução do modelo
    pipeline_model = AKModel()

    # 5. Carrega e executa o modelo
    pipeline_model.run()


if __name__ == "__main__":
    main()
