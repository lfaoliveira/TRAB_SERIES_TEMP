O conjunto de dados utilizado neste trabalho será o **NASA Anomaly Detection Dataset (SMAP/MSL)**, disponibilizado publicamente no Kaggle. O dataset contém séries temporais multivariadas provenientes de telemetria dos satélites **SMAP (Soil Moisture Active Passive)** e **MSL (Mars Science Laboratory - Curiosity Rover)**, sendo amplamente utilizado como benchmark para detecção de anomalias em séries temporais.

Neste trabalho será realizada a **detecção de anomalias (outliers)** e a **modelagem de eventos raros** utilizando técnicas de **Deep Learning**.

### Formato do Dataset

O dataset é composto por múltiplos arquivos contendo séries temporais de sensores de telemetria. Cada arquivo representa um canal (sensor) monitorado ao longo do tempo.

As principais características são:

* Cada arquivo corresponde a uma única série temporal.
* As linhas representam observações consecutivas no tempo (time-steps).
* Cada série possui uma sequência de valores reais medidos por um sensor.
* O conjunto é dividido em treinamento e teste.
* As anomalias estão presentes apenas no conjunto de teste e seus intervalos são fornecidos em um arquivo de rótulos (`labeled_anomalies.csv`), permitindo avaliação supervisionada dos algoritmos.
* As séries possuem comprimentos diferentes e podem apresentar comportamentos bastante distintos entre si.

### Pré-processamento

Antes do treinamento dos modelos poderão ser realizadas etapas de pré-processamento, tais como:

* normalização ou padronização dos valores;
* criação de janelas temporais (sliding windows) para alimentar os modelos de Deep Learning;
* separação entre conjuntos de treinamento, validação e teste.

Embora este dataset normalmente **não possua valores ausentes (NaN)**, deve-se verificar sua integridade antes do treinamento. Caso sejam encontrados valores faltantes, técnicas de imputação poderão ser empregadas com cautela, uma vez que modificações artificiais na série podem alterar ou mascarar o comportamento das anomalias originais, comprometendo a avaliação dos modelos.
