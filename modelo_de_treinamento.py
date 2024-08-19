# Importar as bibliotecas necessárias
import os  # Biblioteca para manipulação de arquivos e diretórios
import numpy as np  # Biblioteca para operações numéricas
from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator  # Ferramenta para processamento e augmentação de imagens
from tensorflow.keras.models import Sequential  # Modelo sequencial para construir redes neurais
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, \
    Input  # Camadas usadas na rede neural
from tensorflow.keras.callbacks import ModelCheckpoint, \
    EarlyStopping  # Ferramentas para monitorar e melhorar o treinamento
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos
import tensorflow as tf  # Biblioteca principal para criar e treinar o modelo de IA

# Definir parâmetros do modelo e do pré-processamento
TAM_IMAGEM = (
    150, 150)  # Tamanho para o qual todas as imagens serão redimensionadas antes de serem alimentadas no modelo
BATCH_SIZE = 32  # Número de imagens processadas por vez durante o treinamento do modelo
EPOCHS = 10  # Número de épocas, ou seja, o número de vezes que o modelo passará por todo o conjunto de dados

# Caminho para a pasta que contém as imagens organizadas em subpastas por classe
caminho_dados = 'imagens_baixadas'

# Criar um gerador de dados de imagem para o treinamento com augmentação (aumento de dados)
datagen_treinamento = ImageDataGenerator(
    rescale=1. / 255,  # Normaliza os valores dos pixels para o intervalo [0, 1]
    rotation_range=40,  # Rotaciona aleatoriamente as imagens em até 40 graus
    width_shift_range=0.2,  # Translada horizontalmente as imagens em até 20% da largura total
    height_shift_range=0.2,  # Translada verticalmente as imagens em até 20% da altura total
    shear_range=0.2,  # Aplica cisalhamento aleatório nas imagens
    zoom_range=0.2,  # Aplica zoom aleatório nas imagens
    horizontal_flip=True,  # Realiza flip horizontal aleatório nas imagens
    fill_mode='nearest'  # Preenche os píxeis vazios após transformação com o valor mais próximo
)

# Criar um gerador de dados de imagem para a validação, apenas com normalização
datagen_validacao = ImageDataGenerator(rescale=1. / 255)

# Criar um gerador para carregar as imagens do diretório de treinamento
gerador_treinamento = datagen_treinamento.flow_from_directory(
    caminho_dados,  # Caminho para o diretório contendo as imagens
    target_size=TAM_IMAGEM,  # Redimensiona as imagens para o tamanho definido
    batch_size=BATCH_SIZE,  # Número de imagens a serem processadas em cada batch
    class_mode='categorical'  # Especifica que a tarefa é de classificação com múltiplas categorias
)

# Criar um gerador para carregar as imagens do diretório de validação
gerador_validacao = datagen_validacao.flow_from_directory(
    caminho_dados,  # Caminho para o diretório contendo as imagens
    target_size=TAM_IMAGEM,  # Redimensiona as imagens para o tamanho definido
    batch_size=BATCH_SIZE,  # Número de imagens a serem processadas em cada batch
    class_mode='categorical'  # Especifica que a tarefa é de classificação com múltiplas categorias
)

# Definir a arquitetura do modelo
modelo = Sequential([
    # Camada de entrada com o formato da imagem, incluindo o tamanho e o número de canais (RGB)
    Input(shape=(TAM_IMAGEM[0], TAM_IMAGEM[1], 3)),

    # Camada convolucional com 32 filtros, cada um com uma janela de 3x3 píxeis
    Conv2D(32, (3, 3), activation='relu'),
    # Camada de pooling para reduzir as dimensões das imagens
    MaxPooling2D((2, 2)),

    # Camada convolucional com 64 filtros
    Conv2D(64, (3, 3), activation='relu'),
    # Camada de pooling
    MaxPooling2D((2, 2)),

    # Camada convolucional com 128 filtros
    Conv2D(128, (3, 3), activation='relu'),
    # Camada de pooling
    MaxPooling2D((2, 2)),

    # Camada convolucional com 128 filtros
    Conv2D(128, (3, 3), activation='relu'),
    # Camada de pooling
    MaxPooling2D((2, 2)),

    # Achatar a saída para passar para camadas densas
    Flatten(),

    # Camada densa com 512 neurônios e ativação ReLU
    Dense(512, activation='relu'),
    # Camada de dropout para evitar overfitting, desativando aleatoriamente 50% dos neurônios
    Dropout(0.5),
    # Camada densa de saída com número de neurônios igual ao número de classes e ativação softmax
    Dense(gerador_treinamento.num_classes, activation='softmax')
])

# Compilar o modelo
modelo.compile(
    optimizer='adam',  # Otimizador Adam para atualizar os pesos do modelo
    loss='categorical_crossentropy',  # Função de perda para classificação múltipla
    metrics=['accuracy']  # Métrica para avaliar o desempenho do modelo
)

# Definir callbacks para monitorar o treinamento
callbacks = [
    # Salvar o modelo com o melhor desempenho em termos de validação
    ModelCheckpoint('modelo_melhor.keras', save_best_only=True),
    # Parar o treinamento se a perda de validação não melhorar após 3 épocas
    EarlyStopping(monitor='val_loss', patience=3)
]

# Treinar o modelo usando os dados de treinamento e validação
historia = modelo.fit(
    gerador_treinamento,  # Dados de treinamento
    epochs=EPOCHS,  # Número de épocas para o treinamento
    validation_data=gerador_validacao,  # Dados de validação
    callbacks=callbacks  # Callbacks para monitoramento
)

# Salvar o modelo final após o treinamento
modelo.save('modelo_final.keras')

# Plotar gráficos de precisão e perda durante o treinamento
plt.figure(figsize=(12, 6))

# Gráfico de precisão
plt.subplot(1, 2, 1)
plt.plot(historia.history['accuracy'], label='Precisão de Treinamento')
plt.plot(historia.history['val_accuracy'], label='Precisão de Validação')
plt.xlabel('Épocas')
plt.ylabel('Precisão')
plt.legend()
plt.title('Precisão do Modelo')

# Gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(historia.history['loss'], label='Perda de Treinamento')
plt.plot(historia.history['val_loss'], label='Perda de Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.title('Perda do Modelo')

# Mostrar os gráficos
plt.show()
