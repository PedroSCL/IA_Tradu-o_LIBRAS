# 🤟 Tradutor de Libras para Áudio com IA

**Disciplina:** Inteligência Artificial  
**Tipo:** Projeto Prático — Protótipo Funcional

---

## Objetivo

Protótipo de sistema baseado em IA que utiliza a câmera do dispositivo para reconhecer, em tempo real, palavras em Língua Brasileira de Sinais (Libras) e convertê-las em áudio (Text-to-Speech).

## Pipeline de IA

```
Câmera → MediaPipe (landmarks) → LSTM (classificação) → Texto → TTS (áudio)
```

## Estrutura do Projeto

```
libras_tradutor/
├── data/
│   ├── raw/                  # Vídeos brutos coletados
│   └── processed/            # Landmarks extraídos (CSV/NPY)
├── src/
│   ├── data_collection/      # Etapa 1: Coleta de dados com câmera
│   ├── preprocessing/        # Etapa 2: Extração de landmarks (MediaPipe)
│   ├── model/                # Etapa 3: Treino da rede LSTM
│   ├── evaluation/           # Etapa 4: Métricas e avaliação
│   └── interface/            # Etapa 5+6: Interface + síntese de voz
├── models/                   # Modelos treinados (.h5 / .keras)
├── notebooks/                # Análises exploratórias
├── reports/                  # Relatório final + figuras
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Como Executar

### 1. Coletar dados
```bash
python src/data_collection/collect_data.py
```

### 2. Pré-processar (extrair landmarks)
```bash
python src/preprocessing/extract_landmarks.py
```

### 3. Treinar o modelo LSTM
```bash
python src/model/train.py
```

### 4. Avaliar o modelo
```bash
python src/evaluation/evaluate.py
```

### 5. Rodar a interface em tempo real
```bash
python src/interface/app.py
```

## Sinais Suportados (vocabulário inicial)

| Sinal | Label |
|-------|-------|
| Olá | ola |
| Obrigado | obrigado |
| Água | agua |
| Preciso de ajuda | ajuda |
| Sim | sim |
| Não | nao |

## Tecnologias

- **MediaPipe** — extração de landmarks das mãos/corpo
- **TensorFlow / Keras** — modelo LSTM
- **OpenCV** — captura de vídeo
- **pyttsx3 / gTTS** — síntese de voz (TTS)
- **scikit-learn** — métricas de avaliação
- **Matplotlib / Seaborn** — visualizações

## Referências

- [IA Libras](https://ialibras.com.br/)
- [MediaPipe Docs](https://mediapipe.readthedocs.io/)
- Hand Talk — ISLR (Isolated Sign Language Recognition)
