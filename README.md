# 🤟 Tradutor de Libras para Áudio com IA

Projeto de Inteligência Artificial capaz de reconhecer sinais em Libras utilizando visão computacional e converter o resultado em áudio em tempo real.

---

# 📌 Objetivo

Desenvolver um protótipo funcional que:

- capture movimentos pela câmera;
- extraia landmarks corporais e das mãos com MediaPipe;
- utilize uma rede neural LSTM para reconhecer sinais;
- converta o resultado em texto e áudio (Text-to-Speech).

---

# 🧠 Pipeline da IA

```text
Câmera → MediaPipe → Landmarks → LSTM → Texto → Áudio
```

---

# 📂 Estrutura do Projeto

```text
libras_tradutor/
├── data/
│   ├── raw/                  # Dados brutos (não incluídos)
│   ├── videos/               # Vídeos originais (não incluídos)
│   └── processed/            # Dados processados (.npy)
│
├── src/
│   ├── data_collection/      # Coleta de dados
│   ├── preprocessing/        # Extração de landmarks
│   ├── model/                # Treinamento da LSTM
│   ├── evaluation/           # Avaliação do modelo
│   └── interface/            # Interface em tempo real
│
├── models/                   # Modelos treinados
├── notebooks/                # Análises exploratórias
├── reports/                  # Relatórios e figuras
├── requirements.txt
└── README.md
```

---

# ⚙️ Requisitos

- Python 3.11+
- Webcam
- Windows/Linux

---

# 🚀 Instalação Completa

## 1. Clone o repositório

```bash
git clone https://github.com/SEUUSUARIO/libras-tradutor.git
```

---

## 2. Entre na pasta do projeto

```bash
cd libras-tradutor
```

---

# 🐍 Criando o Ambiente Virtual (venv)

## Windows (PowerShell)

```bash
python -m venv .venv311
```

Ative:

```bash
.venv311\Scripts\activate
```

---

## Linux / Mac

```bash
python3 -m venv .venv311
source .venv311/bin/activate
```

---

# 📦 Instalar Dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

---
# 📥 Dados Processados

Os arquivos de treinamento não estão incluídos no repositório devido ao tamanho.

Baixe os arquivos abaixo:

- X.npy
- y.npy

Link:
https://drive.google.com/drive/folders/11kP-HvVccio20pZzreGlBPeQHD-F3HJp?usp=sharing

Depois coloque os arquivos em:

```text
data/processed/
```

---

# 🧪 Como Executar

---

## 1. Coletar dados

Captura vídeos/imagens pela câmera.

```bash
python src/data_collection/collect_data.py
```

---

## 2. Extrair landmarks

Converte vídeos/imagens em landmarks numéricos utilizando MediaPipe.

```bash
python src/preprocessing/extract_landmarks.py
```

Os arquivos gerados serão:

```text
data/processed/X.npy
data/processed/y.npy
```

---

## 3. Treinar o modelo LSTM

```bash
python src/model/train.py
```

O modelo treinado será salvo em:

```text
models/
```

---

## 4. Avaliar o modelo

```bash
python src/evaluation/evaluate.py
```

---

## 5. Executar a interface em tempo real

```bash
python src/interface/app.py
```

---

# 🧾 Sinais Suportados

| Sinal | Label |
|-------|-------|
| Olá | ola |
| Obrigado | obrigado |
| Água | agua |
| Preciso de ajuda | ajuda |
| Sim | sim |
| Não | nao |

---

# 🛠 Tecnologias Utilizadas

- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- pyttsx3 / gTTS

---

# 📊 Modelo Utilizado

O projeto utiliza uma rede neural recorrente do tipo LSTM (Long Short-Term Memory), adequada para reconhecimento de padrões temporais em sequências de landmarks extraídos dos sinais em Libras.

---

# 📌 Observações

- As pastas `data/raw/` e `data/videos/` não estão incluídas devido ao tamanho elevado dos arquivos.
- O ambiente virtual `.venv311/` também não é incluído no repositório.
- O modelo pode ser retreinado utilizando novos dados e landmarks.

---

# 📚 Referências

- MediaPipe
- TensorFlow
- OpenCV
- IA Libras
- Hand Talk

---

# 👨‍💻 Autor

Pedro Henrique  
Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial.