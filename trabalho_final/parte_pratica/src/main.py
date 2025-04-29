from data_preparation import DataPreparation

def main():
    pipeline = DataPreparation()

    # 1. Baixar o dataset
    pipeline.download_data()

    # 2. Load metadata
    pipeline.load_metadata()

    # 3. Gerar as imagens dos sinais para o treinamento e teste
    pipeline.signals_to_images()

if __name__ == "__main__":
    main()
