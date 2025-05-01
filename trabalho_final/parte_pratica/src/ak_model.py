import os
import pandas as pd
import autokeras as ak

class AKModel():
    def __init__(self,
                 work_dir: str = "data/physionet",
                 images_dir: str = "ecg/images",
                 csv_dir: str = "ecg/csv",
                 train_subdir: str = "train",
                 test_subdir: str = "test"):
        
        self.work_dir = work_dir
        self.metadata_dir_train = os.path.join(csv_dir, train_subdir)
        self.metadata_dir_test = os.path.join(csv_dir, test_subdir)
        self.images_dir_train = os.path.join(images_dir, train_subdir)
        self.images_dir_test = os.path.join(images_dir, test_subdir)


    def load_model(self):
        pass
        # Cria o ImageClassifier
        #clf = ak.ImageClassifier(overwrite=True, max_trials=2)  # 5 tentativas diferentes de redes automáticas


        # Treina usando o caminho das imagens + label
        #clf.fit(x=df['image_path'].tolist(), y=df['label'].tolist(), epochs=10)


        #clf.fit(x=df['image_path'].tolist(), y=df['label'].tolist(), epochs=10)






        # Depois para testar
        #test_df = pd.read_csv('/content/ecg_labels_3classes_test.csv')
        #predictions = clf.predict(test_df['image_path'].tolist())

    
