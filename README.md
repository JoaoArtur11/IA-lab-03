# Laboratório 3 - Implementando o Decoder

Este projeto organiza a resolução do laboratório em **3 arquivos separados**, um para cada tarefa, além de um arquivo `main.py` que executa tudo em sequência.

## Estrutura dos arquivos

```text
laboratorio_decoder/
├── tarefa1_mascara_causal.py
├── tarefa2_cross_attention.py
├── tarefa3_loop_inferencia.py
├── main.py
├── requirements.txt
└── README.md
```

## O que cada arquivo faz

- `tarefa1_mascara_causal.py`  
  Implementa a máscara causal (look-ahead mask), mostra os scores de atenção e comprova que os tokens futuros ficam com probabilidade `0.0`.

- `tarefa2_cross_attention.py`  
  Implementa a cross-attention entre a saída do encoder e o estado do decoder, usando projeções manuais com `numpy`.

- `tarefa3_loop_inferencia.py`  
  Simula um loop de inferência auto-regressivo, gerando tokens um por um até encontrar o token `<EOS>`.

- `main.py`  
  Executa as três tarefas em sequência.

## Requisitos

Este projeto usa apenas:

- Python 3.x
- `numpy`

## Como instalar as dependências

No terminal, dentro da pasta do projeto, execute:

```bash
pip install -r requirements.txt
```

## Como executar

Depois de instalar as dependências, rode:

```bash
python main.py
```

## Saída esperada

Ao executar o `main.py`, o programa vai:

1. mostrar a criação e aplicação da máscara causal;
2. exibir os shapes principais da cross-attention;
3. simular a geração auto-regressiva de tokens;
4. imprimir a frase final gerada no console.

A saída geral deve seguir essa ideia:

```text
LABORATÓRIO 3 - IMPLEMENTANDO O DECODER
Execução completa das 3 tarefas.

============================================================
TAREFA 1 - IMPLEMENTANDO A MÁSCARA CAUSAL
============================================================

... saída da tarefa 1 ...

============================================================
TAREFA 2 - A PONTE ENCODER-DECODER (CROSS-ATTENTION)
============================================================

... saída da tarefa 2 ...

============================================================
TAREFA 3 - SIMULANDO O LOOP DE INFERÊNCIA AUTO-REGRESSIVO
============================================================

... saída da tarefa 3 ...

Frase final gerada:
eu gosto de numpy
```

## Observações

- O projeto foi feito de forma didática, para facilitar o entendimento.
- Não foram usadas bibliotecas como PyTorch, TensorFlow ou Keras.
- A função `softmax` foi implementada manualmente.
- A geração da terceira tarefa é apenas uma simulação educativa, não um modelo treinado de verdade.

## Sugestão de execução

Para evitar erro de importação, mantenha todos os arquivos na mesma pasta antes de executar o `main.py`.


## Uso de Inteligência Artificial

Durante o desenvolvimento deste projeto, a IA do **ChatGPT** foi utilizada como ferramenta de apoio para:

- auxiliar no entendimento do desafio proposto no laboratório;
- esclarecer conceitos relacionados ao funcionamento do decoder em Transformers;
- ajudar na organização da solução e na estruturação do projeto;
- gerar e revisar o texto deste arquivo **README.md**.
- auxiliar na documentação de código
