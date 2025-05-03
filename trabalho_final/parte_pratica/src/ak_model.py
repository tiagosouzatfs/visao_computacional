import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History

class AKModel():
    def __init__(self,
                 save_images_dir: str = os.path.join("results", "images"),
                 images_dir: str = os.path.join("ecg", "images"),
                 csv_dir: str = os.path.join("ecg", "csv"),
                 train_subdir: str = "train",
                 test_subdir: str = "test"):
        
        self.save_images_dir = save_images_dir
        self.images_dir = images_dir
        self.metadata_dir_train = os.path.join(csv_dir, train_subdir)
        self.metadata_dir_test = os.path.join(csv_dir, test_subdir)
        self.images_dir_train = os.path.join(images_dir, train_subdir)
        self.images_dir_test = os.path.join(images_dir, test_subdir)

    def run(self):

        df_metadata_train = pd.read_csv(os.path.join(self.metadata_dir_train, "df_metadata_train.csv"), sep=";")
        df_metadata_test = pd.read_csv(os.path.join(self.metadata_dir_test, "df_metadata_test.csv"), sep=";")

        df_metadata_train["paths_images_train"] = df_metadata_train["paths_images_train"].astype(str) + ".png"
        df_metadata_test["paths_images_test"] = df_metadata_test["paths_images_test"].astype(str) + ".png"

        def load_images(paths, target_size=(128, 128)):
            return np.array([
                img_to_array(load_img(path, target_size=target_size)) / 255.0
                for path in paths
            ])

        X_train = load_images(df_metadata_train["paths_images_train"])
        X_test = load_images(df_metadata_test["paths_images_test"])

        le = LabelEncoder()
        y_train = le.fit_transform(df_metadata_train["diagnosis"])
        y_test = le.transform(df_metadata_test["diagnosis"])

        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)

        def build_model(base_model_class, input_shape=(128, 128, 3), num_classes=3):
            base_model = base_model_class(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
            x = GlobalAveragePooling2D()(base_model.output)
            output = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=output)
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        for name, base_model in [("EfficientNetB0", EfficientNetB0), ("ResNet50", ResNet50)]:
            print(f"\nTreinando modelo: {name}")
            epocas = 10
            model = build_model(base_model, input_shape=(128, 128, 3), num_classes=y_train_cat.shape[1])
            history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=epocas, batch_size=32)

            # Avaliação
            loss, acc = model.evaluate(X_test, y_test_cat)
            print(f"Acurácia com {name}: {acc:.4f}")

            # Gráficos
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Treino')
            plt.plot(history.history['val_accuracy'], label='Validação')
            plt.title(f'Acurácia - {name}')
            plt.xlabel('Épocas')
            plt.ylabel('Acurácia')
            plt.legend()
            plt.grid(True)
            

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Treino')
            plt.plot(history.history['val_loss'], label='Validação')
            plt.title(f'Loss - {name}')
            plt.xlabel('Épocas')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.grid(True)

            plt.savefig(os.path.join(self.save_images_dir, f"{name}_{epocas}_accuracy_loss.png"))
            plt.close()

            # Matriz de Confusão
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_test, y_pred_classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            plt.figure(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Matriz de Confusão - {name}')
            plt.savefig(os.path.join(self.save_images_dir, f"{name}_{epocas}_confusion_matrix.png"))
            plt.close()
