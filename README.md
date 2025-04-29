# Projeto de IA para Previsão de Produtividade Agrícola - Sprint 2

## Visão Geral

Este projeto, desenvolvido como parte da Sprint 2 do Challenge FIAP, foca na criação de um modelo de Inteligência Artificial para prever a produtividade agrícola (especificamente o rendimento médio em kg/ha). A abordagem envolveu o uso de imagens de satélite para segmentação de áreas de interesse e a integração com dados históricos de produtividade e séries temporais de NDVI para uma fazenda específica (Yrere) no município de Ilhéus, BA. Foram testados múltiplos modelos de regressão para a previsão final de rendimento.

## Estrutura do Repositório

- `/`: Contém este README.md.
- `src/`: Contém o(s) notebook(s) ou script(s) Python com o código completo.
  - `Challenge_Ingredion_Sprint_2.ipynb`
  - `requirements.txt`: Lista das bibliotecas Python necessárias.
- `data/`: Contém os arquivos de dados de entrada.
  - `Producao.csv`: Dados históricos de produtividade (Fonte: IBGE, filtrado/referente a Ilhéus/Fazenda Yrere).
  - `NDVI.csv`: Dados de série temporal NDVI (Fonte: Plataforma SatVeg referente a um ponto na Fazenda Yrere).
  - `test`: Pasta com as imagens de test do treinamento
  - `train`: Pasta com as imagens de treinamento
  - `val`: Pasta com as imagens de validação do treinamento
- `images/`: Contém imagens de exemplo, prints de análises e gráficos de resultados para incluir neste README.
- `output/`: Contém modelos salvos (ex: `best_segmentation_model.pth`).

**Link do Repositório: <https://github.com/Fiap-Team-1tiaor-2024/ingredion-challenge>**

## 1. Preparação e Pré-processamento dos Dados

O processo de preparação dos dados envolveu múltiplas fontes e etapas:

### 1.1. Dados de Imagem e Máscara (Segmentação)

- **Fonte:** Imagens de satélite (`.jpg`) e máscaras de segmentação (`_mask.png`).
- **Carregamento:** Classe `SatelliteDataset` customizada em PyTorch.
- **Pré-processamento:** Conversão RGB/Binário, Transformações (Resize, Augmentation, Normalize) com Albumentations, DataLoaders PyTorch.

### 1.2. Dados Históricos de Produtividade

- **Fonte:** Arquivo `Producao.csv` (anual, 1994-2023, Ilhéus/Yrere).
- **Pré-processamento:** Seleção (`Ano`, `Área colhida`, `Rendimento médio`), Renomeação (`Area_Colhida_ha`, `Rendimento_Medio_kg_ha`).

### 1.3. Dados de NDVI

- **Fonte:** Arquivo `NDVI.csv` (série temporal, ponto de referência, 2000-2025).
- **Pré-processamento:** Leitura com `pd.read_excel`, tratamento de cabeçalho, conversão de `Data` (datetime) e `NDVI` (float), criação da coluna `Ano`.

### 1.4. Integração e Engenharia de Features (Dados Tabulares)

- **Agregação NDVI:** Cálculo de estatísticas anuais (`ndvi_medio`, `ndvi_max`, etc.) e trimestrais (`ndvi_medio_trim_TX`) a partir dos dados de NDVI (filtrados > 0.1 e para anos <= 2023).
- **Features Meteorológicas (Exemplo):** Demonstração da agregação anual e trimestral de dados hipotéticos de precipitação (o código para carregar e agregar dados reais do INMET foi fornecido como exemplo, mas precisa ser implementado com dados reais baixados pelo usuário).
- **Merge:** União dos dataframes de produtividade, NDVI agregado (anual e trimestral) e clima agregado (anual e trimestral - se implementado) usando a coluna `Ano`.
- **Features Lagged:** Criação de features do ano anterior (`_lag1`) para rendimento e área colhida.
- **Limpeza Final:** Remoção de linhas com `NaN` (anos iniciais sem NDVI/lag/clima), resultando no dataframe `df_ready_for_model` (ou `df_final_cleaned`) para modelagem (anos 2001-2023, ou o período resultante após adicionar features climáticas).

## 2. Justificativa das Variáveis Selecionadas

Para o modelo final de previsão de rendimento, as seguintes variáveis foram consideradas/utilizadas:

- **Variável Alvo (y):**
  - `Rendimento_Medio_kg_ha`: Produtividade por unidade de área.
- **Features (X) Potenciais:**
  - `Area_Colhida_ha`: Área colhida no ano corrente.
  - Estatísticas NDVI (Anuais/Trimestrais): `ndvi_medio`, `ndvi_max`, `ndvi_min`, `ndvi_std`, `ndvi_medio_trim_TX`. Indicadores da saúde da vegetação.
  - Features Meteorológicas (Anuais/Trimestrais): Ex: `precip_total_anual`, `temp_max_media_anual`, `precip_total_trim_TX`. Fatores climáticos que influenciam fortemente a agricultura (requer dados do INMET).
  - Features Lagged: `Rendimento_Medio_kg_ha_lag1`, `Area_Colhida_ha_lag1`. Desempenho do ano anterior.
- **Feature de Segmentação (Não Utilizada):** Conforme mencionado, não foi possível integrar uma feature da área segmentada devido à falta de mapeamento temporal das imagens.

A seleção final de features para treinar os modelos (`features` no código) inclui as colunas disponíveis no dataframe final após a integração e engenharia.

## 3. Justificativa dos Modelos e Lógica Preditiva

Dois tipos principais de modelos foram desenvolvidos:

### 3.1. Modelo de Segmentação de Imagens

- **Modelo:** `DeepLabV3+` com backbone `ResNet-50`, pré-treinado.
- **Justificativa:** Estado-da-arte para segmentação semântica, adaptado para classificação binária (cultivo/não-cultivo) para identificar áreas relevantes nas imagens de satélite.
- **Lógica Preditiva:** Aprende padrões visuais para gerar máscaras de probabilidade, avaliado por IoU.

### 3.2. Modelos de Previsão de Rendimento (Comparativo)

- **Modelos Testados:** `Linear Regression`, `Ridge Regression`, `Decision Tree Regressor`, `Random Forest Regressor`, `Gradient Boosting Regressor`.
- **Justificativa:** Para avaliar diferentes abordagens de regressão no dataset disponível (relativamente pequeno após limpeza). Foram incluídos:
  - Modelos Lineares (Linear, Ridge): Baselines simples, bons para relações lineares e interpretação de coeficientes. Ridge adiciona regularização.
  - Árvore de Decisão: Modelo não-linear simples, propenso a overfitting mas útil para entender partições nos dados.
  - Modelos Ensemble (Random Forest, Gradient Boosting): Combinam múltiplas árvores para melhorar a robustez, capturar não-linearidades e interações, geralmente com melhor desempenho preditivo. RandomForest é menos sensível a hiperparâmetros, Gradient Boosting pode ser mais potente mas requer mais tuning.
- **Lógica Preditiva:** Cada modelo aprende a mapear as features de entrada (área, NDVI, clima, lags) para uma previsão do valor contínuo do `Rendimento_Medio_kg_ha`, usando suas respectivas lógicas internas (linear, partições de árvore, combinação de árvores).
- **Avaliação:** Comparação baseada em RMSE, MAE e R² no conjunto de teste (separado temporalmente), além da análise de importância/coeficientes das features.

## 4. Análises Exploratórias e Estatísticas

- **Análise NDVI:** Estatísticas descritivas calculadas, série temporal plotada (opcional).
- **Análise de Correlação:** Matriz de correlação calculada e visualizada para features e alvo no dataset final.
  ![Heatmap](/images//heatmap.png)

- _Outras análises:_  
  **NDVI:**  
  ![NDVI](/images/ndvi.png)  
  **Pairplot:**  
  ![pairplot](/images/pairplot.png)

## 5. Resultados e Justificativa Técnica

### 5.1. Resultados da Segmentação

- Modelo DeepLabV3+ treinado por `200` épocas.
- Melhor IoU na validação: `0.4546`.
- **IoU médio no teste: `0.4200`**.

  - **Visualização de Exemplo:**
    ![predicao_de_segmentacao](/images/predicao_de_segmentacao.png)

### 5.2. Resultados da Previsão de Rendimento (Comparativo de Modelos)

- Múltiplos modelos de regressão foram treinados nos dados de `2000` a `2023` e testados nos anos `2016` a `2023`.
- **Comparativo de Métricas (Conjunto de Teste):**
  

- **Interpretação:** Observou-se que  todos os modelos apresentaram R² baixo ou negativo, indicando dificuldade em generalizar. O modelo SVR teve o melhor desempenho relativo, mas ainda assim não foi satisfatório chegando a valor de R2 = -0.989003. As possíveis causas incluem o dataset pequeno, a falta de features importantes (como dados meteorológicos detalhados, por exemplo) ou mudanças no comportamento da série temporal.
  - **Gráfico Comparativo Previsto vs Real:**
    ![comparativo_previsto](/images/comparativo_previsto.png)
- **Importância das Features / Coeficientes:** A análise mostrou que [Resumir principais features, ex: features lagged tiveram maior importância para os modelos de árvore, enquanto features X foram mais relevantes para modelos Y].
  ![linear_regression](/images/linear_regression.png)
  ![ridge_regression](/images/ridge_regression.png)
  ![decision_tree](/images/decision_tree.png)
  ![random_forrest](/images/random_forrest.png)
  ![gradient_boost](/images/gradient_boost.png)
  ![xgboost](/images/xgboost.png)

- **Conclusão Técnica:** O projeto implementou com sucesso um pipeline completo, desde a segmentação de imagens até a modelagem de previsão de rendimento com comparação de algoritmos. No entanto, a capacidade preditiva do modelo de rendimento foi limitada com os dados e features atuais, destacando a importância de enriquecimento de dados (clima, manejo) e potencialmente de um histórico mais longo para melhorar a acurácia em futuras iterações. A análise comparativa dos modelos ajuda a entender quais abordagens podem ser mais promissoras com dados melhores.

## 6. Como Executar (Usando Google Colab)

Este projeto foi desenvolvido e testado primariamente no Google Colab. Siga os passos abaixo para executá-lo:

1. **Abra o Notebook no Google Colab:**

   - Acesse o repositório do projeto no GitHub: [https://github.com/Fiap-Team-1tiaor-2024/ingredion-challenge](https://github.com/Fiap-Team-1tiaor-2024/ingredion-challenge)
   - Navegue até a pasta que contém o arquivo `.ipynb` principal.
   - Clique no arquivo do notebook (ex: `Challenge_Ingredion_Sprint_2.ipynb`).
   - No topo da visualização do notebook no GitHub, clique no emblema "Open in Colab".

2. **Prepare o Ambiente Colab:**

   - **Monte seu Google Drive:** Execute a primeira célula do notebook para montar seu Google Drive e autorize o acesso.
   - **Instale Dependências (se necessário):** Verifique a primeira célula por comandos `!pip install ...`.

3. **Configure os Caminhos dos Arquivos:**

   - **MUITO IMPORTANTE:** Na primeira célula de código, **AJUSTE** as variáveis de caminho (`DRIVE_BASE_PATH`, `PROD_CSV_PATH`, `NDVI_FILE_PATH`, `SAVE_DIR`, etc.) para que apontem para os locais corretos dos seus dados no **seu Google Drive**.

4. **Execute as Células:**

   - Execute as células do notebook sequencialmente ("Ambiente de execução" -> "Executar tudo" ou célula por célula).
   - Acompanhe a saída de cada célula. O treinamento de segmentação pode demorar.

5. **Resultados:**
   - Gráficos e métricas serão exibidos nas saídas das células.
   - O modelo de segmentação treinado será salvo no `SAVE_DIR` no seu Drive.

## 7. Autores

- Gabriela da Cunha Rocha - <RM561041@fiap.com.br>
- Gustavo Segantini Rossignolli - <RM560111@fiap.com.br>
- Vitor Lopes Romão - <RM559858@fiap.com.br>

---
