import numpy as np

# softmax manual
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_attention(encoder_out, decoder_state):
    # pega as dimensões das entradas
    batch_size, seq_enc, d_model = encoder_out.shape
    _, seq_dec, _ = decoder_state.shape

    # pesos aleatorios (so pra simular, em producao seriam aprendidos)
    np.random.seed(123)
    W_q = np.random.randn(d_model, d_model) * 0.02
    W_k = np.random.randn(d_model, d_model) * 0.02
    W_v = np.random.randn(d_model, d_model) * 0.02

    # projecoes lineares
    # Q vem do decoder, K e V vem do encoder
    Q = decoder_state @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v

    # calcula os scores e normaliza pela raiz de d_model
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
    pesos = softmax(scores, axis=-1)
    saida = pesos @ V

    return Q, K, V, scores, pesos, saida


def run_task_2():
    print("\n" + "=" * 60)
    print("TAREFA 2 - A PONTE ENCODER-DECODER (CROSS-ATTENTION)")
    print("=" * 60)

    np.random.seed(7)

    # simulando uma entrada: 1 sequencia de 10 tokens, dimensao 512
    encoder_output = np.random.randn(1, 10, 512)
    decoder_state = np.random.randn(1, 4, 512)

    Q, K, V, scores, pesos, saida = cross_attention(encoder_output, decoder_state)

    print(f"\nShape encoder_output: {encoder_output.shape}")
    print(f"Shape decoder_state:  {decoder_state.shape}")
    print(f"Shape Q:              {Q.shape}")
    print(f"Shape K:              {K.shape}")
    print(f"Shape V:              {V.shape}")
    print(f"Shape scores:         {scores.shape}")
    print(f"Shape pesos:          {pesos.shape}")
    print(f"Shape saida:          {saida.shape}")

    print("\nPesos de atenção da primeira posição do decoder (sobre os 10 tokens do encoder):")
    print(np.round(pesos[0, 0], 4))

    print("\nSoma dos pesos (tem que dar 1.0):")
    print(np.sum(pesos[0, 0]))

    print("\nInterpretação:")
    print("O decoder_state gera as Queries (Q).")
    print("A saída do encoder gera as Keys (K) e os Values (V).")
    print("Cada posição do decoder pode olhar todos os tokens do encoder.")
    print("Por isso não tem máscara causal aqui.")