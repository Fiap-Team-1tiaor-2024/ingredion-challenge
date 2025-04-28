# Projeto de IA para Previsão de Produtividade Agrícola - Sprint 2

## Visão Geral

Este projeto, desenvolvido como parte da Sprint 2 do Challenge FIAP, foca na criação de um modelo de Inteligência Artificial para prever a produtividade agrícola (especificamente o rendimento médio em kg/ha). A abordagem envolveu o uso de imagens de satélite para segmentação de áreas de interesse e a integração com dados históricos de produtividade e séries temporais de NDVI para uma fazenda específica (Yrere) no município de Ilhéus, BA.

## Estrutura do Repositório

* `/`: Contém este README.md.
* `src/`: Contém o(s) notebook(s) ou script(s) Python com o código completo.
    * `Challenge_Ingredion_Sprint_2.ipynb`
    * `requirements.txt`: Lista das bibliotecas Python necessárias.
* `data/`: Contém os arquivos de dados de entrada.
    * `Producao.csv`: Dados históricos de produtividade (Fonte: IBGE, filtrado/referente a Ilhéus/Fazenda Yrere).
    * `NDVI.csv`: Dados de série temporal NDVI (Fonte: Plataforma SatVeg referente a um ponto na Fazenda Yrere).
    * `test`: Pasta com as imagens de test do treinamento
    * `train`: Pasta com as imagens de treinamento
    * `val`: Pasta com as imagens de validação do treinamento
* `images/`: Contém imagens de exemplo, prints de análises e gráficos de resultados para incluir neste README.
* `output/`: Contém modelos salvos (ex: `best_segmentation_model.pth`).


## 1. Preparação e Pré-processamento dos Dados

O processo de preparação dos dados envolveu múltiplas fontes e etapas:

### 1.1. Dados de Imagem e Máscara (Segmentação)

* **Fonte:** Imagens de satélite (formato `.jpg`) e máscaras de segmentação binárias (`_mask.png`) representando áreas de interesse (presumivelmente cultivo).
* **Carregamento:** Foi criada uma classe `SatelliteDataset` customizada em PyTorch para carregar pares de imagem/máscara. Verificações foram implementadas para garantir a existência de máscaras correspondentes e a leitura correta dos arquivos.
* **Pré-processamento:**
    * As imagens foram convertidas para RGB.
    * As máscaras (tons de cinza) foram convertidas para formato binário (0.0 ou 1.0).
    * **Transformações (Albumentations):**
        * *Treino:* Redimensionamento (512x512), Aumentos de dados (Flip Horizontal/Vertical, Brilho/Contraste Aleatório), Normalização (médias/std da ImageNet).
        * *Validação/Teste:* Redimensionamento (512x512), Normalização.
    * **DataLoaders:** Foram criados DataLoaders do PyTorch para treino, validação e teste, com tratamento para batches inválidos (`collate_fn`).

### 1.2. Dados Históricos de Produtividade

* **Fonte:** Arquivo `Producao.csv`, contendo dados anuais (1994-2023) agregados para a Fazenda Yrere / Município de Ilhéus.
* **Pré-processamento:**
    * Carregamento com Pandas.
    * Seleção das colunas relevantes: `Ano`, `Área colhida (Hectares)` e `Rendimento médio da produção (Quilogramas por Hectare)`.
    * Renomeação das colunas para facilitar o uso (`Area_Colhida_ha`, `Rendimento_Medio_kg_ha`).
    * Conversão da coluna `Ano` para tipo inteiro.

### 1.3. Dados de NDVI

* **Fonte:** Arquivo `NDVI.csv`, contendo uma série temporal de valores NDVI para um ponto de referência na área de estudo (2000-2025).
* **Pré-processamento:**
    * Carregamento com Pandas, tratando problemas de cabeçalho no arquivo CSV (usando `header=2` e renomeação manual das colunas para `Data` e `NDVI`).
    * Remoção de linhas/colunas irrelevantes ou vazias.
    * Conversão da coluna `Data` para o formato datetime (dd/mm/yyyy).
    * Conversão da coluna `NDVI` para formato numérico (float), tratando vírgulas como separadores decimais.
    * Remoção de linhas com datas ou valores NDVI inválidos.
    * Criação da coluna `Ano` extraída da data.

### 1.4. Integração e Engenharia de Features (Dados Tabulares)

* **Agregação NDVI:** Os dados de NDVI foram filtrados para o período correspondente aos dados de produtividade (até 2023) e valores muito baixos (possivelmente ruído, NDVI <= 0.1) foram removidos. Em seguida, foram agrupados por `Ano` para calcular estatísticas anuais: `ndvi_medio`, `ndvi_max`, `ndvi_min`, `ndvi_std`, `ndvi_count`.
* **Merge:** O DataFrame de produtividade (`df_prod`) foi unido (merged) com o DataFrame de NDVI agregado (`df_ndvi_agg`) usando a coluna `Ano`. Um `left merge` foi utilizado para manter todos os anos da base de produtividade.
* **Features Lagged:** Foram criadas features baseadas no ano anterior (`lag1`) para o rendimento médio e a área colhida, visando capturar dependências temporais.
* **Limpeza Final:** Linhas contendo valores `NaN` (resultantes do merge para anos sem NDVI e da criação das features lagged) foram removidas, resultando em um DataFrame final (`df_final_cleaned`) cobrindo os anos de 2001 a 2023, pronto para a modelagem.

## 2. Justificativa das Variáveis Selecionadas

Para o modelo final de previsão de rendimento, as seguintes variáveis foram selecionadas:

* **Variável Alvo (y):**
    * `Rendimento_Medio_kg_ha`: Representa a produtividade por unidade de área, uma métrica padrão e normalizada para avaliação de desempenho agrícola.
* **Features (X):**
    * `Area_Colhida_ha`: A área efetivamente colhida pode influenciar a logística e potencialmente se correlacionar com fatores que afetam o rendimento médio.
    * `ndvi_medio`, `ndvi_max`, `ndvi_min`, `ndvi_std`: Estatísticas anuais do NDVI (filtrado > 0.1). O NDVI é um forte indicador da saúde e densidade da vegetação. Esperava-se que a média anual, o pico, a variabilidade e o mínimo durante o período de crescimento pudessem se correlacionar com o rendimento final. A agregação anual foi uma necessidade devido à estrutura dos dados de produtividade disponíveis.
    * `ndvi_count`: Número de leituras válidas de NDVI no ano, pode indicar a qualidade/confiabilidade das estatísticas de NDVI.
    * `Rendimento_Medio_kg_ha_lag1`: O rendimento do ano anterior é frequentemente um forte preditor do rendimento atual devido a fatores persistentes (solo, manejo, clima local).
    * `Area_Colhida_ha_lag1`: A área colhida no ano anterior pode influenciar decisões de plantio e recursos no ano atual.
* **Feature de Segmentação (Não Utilizada):**
    * Embora um modelo de segmentação tenha sido treinado com sucesso (ver seção 3.1), **não foi possível** criar uma feature anual baseada na área segmentada média. Isso ocorreu devido à **impossibilidade de mapear as imagens de satélite individuais (usadas na segmentação) a anos específicos (2001-2023)** para a Fazenda Yrere. Esta limitação impediu a integração direta dos resultados da segmentação no modelo de previsão de rendimento.

## 3. Justificativa dos Modelos e Lógica Preditiva

Dois modelos principais foram desenvolvidos:

### 3.1. Modelo de Segmentação de Imagens

* **Modelo Escolhido:** `DeepLabV3+` com backbone `ResNet-50`, pré-treinado na ImageNet.
    * **Justificativa:** É uma arquitetura estado-da-arte para segmentação semântica, eficaz na captura de contextos multi-escala em imagens. O uso de pesos pré-treinados (transfer learning) acelera o treinamento e melhora o desempenho, especialmente com datasets menores.
* **Adaptação:** A camada classificadora final foi substituída por uma convolução 1x1 com 1 canal de saída, adequada para a tarefa de segmentação binária (identificar uma única classe de interesse vs fundo).
* **Lógica Preditiva:** O modelo aprende a associar padrões de pixels e texturas nas imagens de satélite a uma máscara binária que indica a presença da área de interesse (provavelmente cultivo). A função de perda utilizada foi `BCEWithLogitsLoss`, adequada para segmentação binária.
* **Avaliação:** O desempenho foi avaliado usando a métrica Intersection over Union (IoU), que mede a sobreposição entre a máscara prevista e a máscara real.

### 3.2. Modelo de Previsão de Rendimento

* **Modelo Escolhido:** `RandomForestRegressor` do Scikit-learn.
    * **Justificativa:** Florestas Aleatórias são robustas, lidam bem com relações não-lineares entre features e o alvo, capturam interações entre features e são menos sensíveis à multicolinearidade (observada entre as features NDVI) e à escala das features em comparação com modelos lineares. Adequado para dados tabulares com um número moderado de amostras.
* **Lógica Preditiva:** O modelo combina as previsões de múltiplas árvores de decisão treinadas em subconjuntos dos dados e das features. Ele aprende a mapear as combinações das features de entrada (área colhida, estatísticas NDVI, valores do ano anterior) para uma previsão do valor contínuo do `Rendimento_Medio_kg_ha`.
* **Avaliação:** O desempenho foi avaliado usando RMSE (erro na unidade do alvo - kg/ha), MAE (erro absoluto médio) e R² (proporção da variância explicada). A divisão treino/teste foi feita temporalmente para simular um cenário de previsão real.

## 4. Análises Exploratórias e Estatísticas

* **Análise NDVI:** Foram calculadas estatísticas descritivas para a série temporal de NDVI após a limpeza.
* **Análise de Correlação:** Uma matriz de correlação foi calculada para as features e a variável alvo no DataFrame final (`df_final_cleaned`).

    * *Observação:* As correlações lineares entre as features disponíveis e o rendimento médio foram fracas. No entanto, multicolinearidade foi observada entre as features NDVI.
    * ![Heatmap](/images/Heatmap.png)

    * *Outras análises (adicionar conforme realizado):*
    * ![Análise de Série Temporal NDVI](/images/image.png)
    * ![Pairplot](/images/Pairplot.png)

## 5. Resultados e Justificativa Técnica

### 5.1. Resultados da Segmentação

* O modelo DeepLabV3+ foi treinado por `10` épocas.
* A melhor métrica IoU obtida no conjunto de **validação** foi `0.3583`.
* No conjunto de **teste**, o modelo alcançou um **IoU médio de `[Valor do Test IoU]`**.

    * **Visualização de Exemplo:**
        ![Imagem Original vs Máscara Real vs Predição](/images/imagemreal_mascara_real_predicao.png)

### 5.2. Resultados da Previsão de Rendimento

* O modelo RandomForestRegressor foi treinado nos dados de 2001 a 2024 e testado nos anos 2013 a 2023.
* **Métricas de Avaliação (Conjunto de Teste):**
    * RMSE: `37.9532` kg/ha
    * MAE: `29.9620` kg/ha
    * **R²: `-0.3796`**
* **Interpretação:** O valor do R² foi **negativo (`-0.3796`)**, indicando que o modelo performou pior do que um modelo ingênuo que previsse apenas a média do rendimento para o período de teste. Isso sugere que o modelo **não generalizou bem** para os anos não vistos, possivelmente devido ao pequeno tamanho do dataset de treino (2001-2023), overfitting, ou falta de features preditivas suficientes para capturar a dinâmica do rendimento nos anos de teste.
    * **Gráfico Previsto vs Real:**
    ![Rendimento Previsto vs Real](/images/rendimento_previsto_real.png)
* **Importância das Features:** As features mais importantes identificadas pelo RandomForest foram o rendimento e a área colhida do ano anterior, seguidas pela área colhida atual. As features de NDVI tiveram menor importância relativa neste modelo.
    * `[Inserir Gráfico: Importância das Features aqui]`![Importância das Features](/images/importancia_das_features.png)

* **Conclusão Técnica:** Embora o fluxo de trabalho completo tenha sido implementado, o modelo final de previsão de rendimento apresentou limitações significativas em sua capacidade preditiva para o período de teste com as features disponíveis. Recomenda-se explorar features adicionais (meteorologia detalhada, dados de manejo), modelos alternativos, ou obter um histórico de dados mais longo para melhorar a performance. A limitação na integração da feature de segmentação também impactou o potencial do modelo.

## 6. Como Executar (Usando Google Colab)

Este projeto foi desenvolvido e testado primariamente no Google Colab. Siga os passos abaixo para executá-lo:

1.  **Abra o Notebook no Google Colab:**
    * Acesse o repositório do projeto no GitHub: [https://github.com/Fiap-Team-1tiaor-2024/ingredion-challenge](https://github.com/Fiap-Team-1tiaor-2024/ingredion-challenge)
    * Navegue até a pasta `notebooks/` ou `scripts/` (ou onde quer que o arquivo `.ipynb` principal esteja localizado).
    * Clique no arquivo do notebook (ex: `Challenge_Ingredion_Sprint_2.ipynb`).
    * No topo da visualização do notebook no GitHub, clique no emblema "Open in Colab".

2.  **Prepare o Ambiente Colab:**
    * **Monte seu Google Drive:** A primeira célula do notebook contém código para montar seu Google Drive. Execute esta célula e autorize o acesso quando solicitado. Isso é essencial para carregar seus dados e salvar o modelo treinado.
    * **Instale Dependências (se necessário):** A primeira célula também pode conter comandos `!pip install ...` (geralmente comentados). O Colab já vem com muitas bibliotecas pré-instaladas (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `opencv`, `torch`, `torchvision`), mas se alguma específica estiver faltando ou precisar de uma versão diferente, descomente e execute os comandos `!pip install`. A biblioteca `openpyxl` pode ser necessária para ler o arquivo `.xlsx`: `!pip install openpyxl`.

3.  **Configure os Caminhos dos Arquivos:**
    * **MUITO IMPORTANTE:** Na primeira célula de código (onde estão as configurações e imports), localize as variáveis de caminho como `DRIVE_BASE_PATH`, `PROD_CSV_PATH`, `NDVI_FILE_PATH`, `SAVE_DIR`, etc.
    * **AJUSTE** esses caminhos para que correspondam **exatamente** à estrutura de pastas e aos nomes dos arquivos dentro do **seu Google Drive**, onde você salvou os dados (imagens, máscaras, CSV, XLSX) e onde deseja salvar os resultados (como o modelo treinado).
    * Exemplo: Se seus dados estão em uma pasta chamada "Challenge_Sprint2" dentro do seu "Meu Drive", o `DRIVE_BASE_PATH` seria algo como:
      ```python
      DRIVE_BASE_PATH = "/content/drive/MyDrive/Challenge_Sprint2/data"
      ```

4.  **Execute as Células:**
    * Execute as células do notebook sequencialmente, uma após a outra, clicando no botão "Play" de cada célula ou usando "Ambiente de execução" -> "Executar tudo".
    * Acompanhe a saída de cada célula para verificar se há erros ou avisos.
    * **Atenção:** O treinamento do modelo de segmentação (Célula 6) pode demorar bastante tempo, dependendo da disponibilidade da GPU no Colab e do número de épocas (`NUM_EPOCHS_SEGMENTATION`).

5.  **Resultados:**
    * Os gráficos (Heatmap, Previsto vs Real, etc.) serão exibidos no output das células correspondentes. Salve-os como imagens para incluir neste README.
    * O melhor modelo de segmentação será salvo no caminho especificado pela variável `BEST_MODEL_PATH` no seu Google Drive.
    * As métricas de avaliação dos modelos serão impressas na saída das células de avaliação.

## 7. Autores

* Gabriela da Cunha Rocha - RM561041@fiap.com.br
* Gustavo Segantini Rossignolli - RM560111@fiap.com.br
* Vitor Lopes Romão - RM559858@fiap.com.br
---
