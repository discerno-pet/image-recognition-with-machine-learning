import \
    os  # Importa o módulo 'os' para interagir com o sistema de arquivos, permitindo navegar em diretórios e manipular arquivos.
from PIL import \
    Image  # Importa a classe 'Image' da biblioteca 'PIL' (Pillow) para manipulação de imagens, como abrir e exibir imagens.
import \
    matplotlib.pyplot as plt  # Importa 'pyplot' da biblioteca 'matplotlib' para exibir gráficos e imagens em uma interface gráfica.


def exibir_imagens_por_raca(raça):
    """
    Exibe imagens de uma raça específica armazenadas em uma subpasta.

    Parameters:
    raça (str): Nome da raça para exibir as imagens.
    """
    # Lista todas as subpastas na pasta 'imagens_baixadas'.
    # 'os.listdir()' lista todos os arquivos e pastas no diretório especificado.
    # 'os.path.isdir()' verifica se o caminho é um diretório.
    pastas = [pasta for pasta in os.listdir('imagens_baixadas') if
              os.path.isdir(os.path.join('imagens_baixadas', pasta))]

    # Normaliza o nome da raça fornecido pelo usuário convertendo para minúsculas para comparação consistente.
    raça_normalizada = raça.lower()

    # Inicializa a variável 'pasta_correspondente' como None.
    pasta_correspondente = None

    # Tenta encontrar uma pasta correspondente ao nome da raça fornecido.
    # Compara o nome normalizado da raça com os nomes das pastas.
    for pasta in pastas:
        if raça_normalizada == pasta.lower():
            pasta_correspondente = pasta
            break

    # Se não encontrar uma pasta correspondente, informa o usuário e exibe as pastas disponíveis.
    if not pasta_correspondente:
        print(f"Pasta para a raça '{raça}' não encontrada.")
        print("Pastas disponíveis:")
        for pasta in pastas:
            print(f"- {pasta}")
        return  # Termina a execução da função, pois não há imagens para exibir.

    # Define o caminho para o diretório onde as imagens da raça estão armazenadas.
    diretorio = os.path.join('imagens_baixadas', pasta_correspondente)

    # Lista todos os arquivos no diretório da raça e filtra apenas arquivos regulares.
    # 'os.path.isfile()' verifica se o caminho é um arquivo regular.
    arquivos_imagem = [f for f in os.listdir(diretorio) if os.path.isfile(os.path.join(diretorio, f))]

    # Verifica se a lista de arquivos de imagem está vazia.
    if not arquivos_imagem:
        print(f"Nenhuma imagem encontrada para a raça '{pasta_correspondente}'.")
        return  # Termina a execução da função, pois não há imagens para exibir.

    # Determina o número de imagens a serem exibidas. Limita a 5 imagens ou o número total disponível, o que for menor.
    num_imagens = min(len(arquivos_imagem), 5)

    # Cria uma figura com subplots para exibir as imagens.
    # 'figsize' define o tamanho da figura em polegadas.
    fig, axes = plt.subplots(nrows=1, ncols=num_imagens, figsize=(15, 5))

    # Se há apenas uma imagem, 'axes' é um objeto único, não uma lista.
    # Neste caso, transforma 'axes' em uma lista para garantir a iterabilidade.
    if len(arquivos_imagem) == 1:
        axes = [axes]

    # Itera sobre os subplots (axes) e arquivos de imagem.
    for ax, arquivo in zip(axes, arquivos_imagem):
        # Constrói o caminho completo para o arquivo de imagem.
        caminho_imagem = os.path.join(diretorio, arquivo)
        # Abre a imagem usando PIL.
        img = Image.open(caminho_imagem)
        # Exibe a imagem no subplot.
        ax.imshow(img)
        # Define o título do subplot como o nome do arquivo da imagem.
        ax.set_title(arquivo)
        # Remove os eixos do subplot para uma visualização mais limpa.
        ax.axis('off')

    # Exibe a figura com todos os subplots.
    plt.show()


# Solicita ao usuário o nome da raça das imagens que deseja exibir.
raça = input("Digite o nome da raça que deseja exibir: ")
# Chama a função para exibir imagens da raça escolhida pelo usuário.
exibir_imagens_por_raca(raça)
