import numpy as np

# softmax manual
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# cria a mascara causal: posições futuras recebem -inf
def criar_mascara_causal(seq_len):
    # triangulo superior vira -inf, o resto fica 0
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

def run_task_1():
    print("=" * 60)
    print("TAREFA 1 - IMPLEMENTANDO A MÁSCARA CAUSAL")
    print("=" * 60)

    np.random.seed(42)
    seq_len = 4
    d_k = 4

    # matrizes Q e K aleatorias pra testar
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    # calcula os scores de atencao
    scores = Q @ K.T

    mascara = criar_mascara_causal(seq_len)

    # aplica a mascara somando -inf nas posicoes futuras
    scores_mascarados = scores + mascara

    # softmax nos scores mascarados
    probs = softmax(scores_mascarados, axis=-1)

    print("\nQ:")
    print(np.round(Q, 3))
    print("\nK:")
    print(np.round(K, 3))
    print("\nScores brutos (Q @ K^T):")
    print(np.round(scores, 3))
    print("\nMáscara causal:")
    print(mascara)
    print("\nScores com máscara aplicada:")
    print(scores_mascarados)
    print("\nProbabilidades após softmax:")
    print(np.round(probs, 4))

    # verifica se as posicoes futuras viraram 0
    print("\nVerificando posições futuras:")
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                print(f"  posição [{i},{j}] (futuro) -> {probs[i, j]:.4f}")

    print("\nInterpretação:")
    print("As posições futuras receberam -inf antes do softmax.")
    print("Por isso a probabilidade nessas posições ficou 0.")
    print("Assim o decoder não consegue 'ver' tokens que ainda não foram gerados.")