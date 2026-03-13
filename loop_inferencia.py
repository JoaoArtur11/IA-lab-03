import numpy as np

def softmax(x, axis=-1):
    """
    Softmax implementado manualmente.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Vocabulário fictício simples
VOCAB = {
    0: "<PAD>",
    1: "<BOS>",
    2: "<EOS>",
    10: "eu",
    11: "gosto",
    12: "de",
    13: "estudar",
    14: "numpy",
    15: "transformers",
    16: "muito"
}

ID_TO_TOKEN = VOCAB
TOKEN_TO_ID = {token: idx for idx, token in VOCAB.items()}
VOCAB_SIZE = 10000

def mock_generate_next_token(current_sequence, encoder_out):
    """
    Simula a geração do próximo token pelo decoder.
    Retorna um vetor de probabilidades de tamanho 10000.
    """
    logits = np.full(VOCAB_SIZE, -10.0)

    # Uso simbólico do encoder_out, só para não ficar totalmente solto
    encoder_signal = float(np.mean(encoder_out))

    # Regras simples para simular a geração de uma frase
    if current_sequence == [TOKEN_TO_ID["<BOS>"]]:
        logits[TOKEN_TO_ID["eu"]] = 8.0
        logits[TOKEN_TO_ID["gosto"]] = 1.0

    elif current_sequence[-1] == TOKEN_TO_ID["eu"]:
        logits[TOKEN_TO_ID["gosto"]] = 8.0
        logits[TOKEN_TO_ID["estudar"]] = 3.0

    elif current_sequence[-1] == TOKEN_TO_ID["gosto"]:
        logits[TOKEN_TO_ID["de"]] = 8.0
        logits[TOKEN_TO_ID["muito"]] = 2.0

    elif current_sequence[-1] == TOKEN_TO_ID["de"]:
        if encoder_signal >= 0:
            logits[TOKEN_TO_ID["numpy"]] = 8.0
            logits[TOKEN_TO_ID["transformers"]] = 5.0
        else:
            logits[TOKEN_TO_ID["transformers"]] = 8.0
            logits[TOKEN_TO_ID["numpy"]] = 5.0

    elif current_sequence[-1] in [TOKEN_TO_ID["numpy"], TOKEN_TO_ID["transformers"]]:
        logits[TOKEN_TO_ID["<EOS>"]] = 8.0
        logits[TOKEN_TO_ID["muito"]] = 2.0

    elif current_sequence[-1] == TOKEN_TO_ID["muito"]:
        logits[TOKEN_TO_ID["<EOS>"]] = 8.0

    else:
        logits[TOKEN_TO_ID["<EOS>"]] = 8.0

    return softmax(logits)

def run_task_3():
    print("\n" + "=" * 60)
    print("TAREFA 3 - SIMULANDO O LOOP DE INFERÊNCIA AUTO-REGRESSIVO")
    print("=" * 60)

    np.random.seed(99)
    encoder_out = np.random.randn(1, 10, 512)

    current_sequence = [TOKEN_TO_ID["<BOS>"]]

    print("\nSequência inicial:")
    print([ID_TO_TOKEN[t] for t in current_sequence])

    while True:
        probs = mock_generate_next_token(current_sequence, encoder_out)

        # Escolha do próximo token usando argmax
        next_token_id = int(np.argmax(probs))
        current_sequence.append(next_token_id)

        print("\nToken gerado:", ID_TO_TOKEN.get(next_token_id, f"<ID {next_token_id}>"))
        print("Top 5 probabilidades:")

        top5_ids = np.argsort(probs)[-5:][::-1]
        for idx in top5_ids:
            nome = ID_TO_TOKEN.get(int(idx), f"<ID {int(idx)}>")
            print(f"  {nome:15s} -> {probs[idx]:.6f}")

        # Para assim que gerar o token de fim
        if next_token_id == TOKEN_TO_ID["<EOS>"]:
            break

    # Remove tokens especiais para formar a frase final
    final_tokens = [
        ID_TO_TOKEN[t]
        for t in current_sequence
        if t not in [TOKEN_TO_ID["<BOS>"], TOKEN_TO_ID["<EOS>"], TOKEN_TO_ID["<PAD>"]]
    ]

    final_sentence = " ".join(final_tokens)

    print("\nSequência final de IDs:")
    print(current_sequence)

    print("\nFrase final gerada:")
    print(final_sentence)

    print("\nInterpretação:")
    print("A sequência foi construída token por token.")
    print("A cada passo, o modelo simulado escolheu o próximo token.")
    print("Quando apareceu o token <EOS>, a geração foi encerrada.")
    print("Isso representa de forma didática o processo auto-regressivo.")