"""
本当に最小のAI - たった1層のニューラルネットワーク (numpy版)

仕組み: 「直前の1文字」から「次の1文字」を予測するだけのシンプルなモデル。
PyTorch を使わず、numpy だけで forward (順伝播) / backward (逆伝播) / SGD (学習) を手書き実装。
全体の流れ: データ準備 → 予測 → ハズレ度合い計算 → 重み修正 → 繰り返し → 文章生成
"""

import numpy as np

# --- 学習データ ---
text = "ももたろうはももからうまれた"

# --- 文字と数値の対応表 ---
chars = sorted(set(text))
vocab = len(chars)  # 文字の種類数
to_i = {c: i for i, c in enumerate(chars)}
to_c = {i: c for i, c in enumerate(chars)}

print(f"文字数: {len(text)}  語彙: {vocab}種 {chars}")

# --- 学習データを作る: (入力1文字, 正解=次の1文字) のペア ---
X = np.array([to_i[c] for c in text[:-1]])  # 入力: (13,)
Y = np.array([to_i[c] for c in text[1:]])   # 正解: (13,)
N = len(X)  # サンプル数 = 13

# --- パラメータ初期化 ---
# 最初はデタラメな値でOK。学習を繰り返すことで良い値になっていく。

# np.random.randn(m, n): 標準正規分布 N(0,1) から m×n の乱数行列を生成
# (= 0 を中心にバラつきのあるランダムな数。ほとんどが -3〜3 に収まる)
emb_weight = np.random.randn(vocab, 8)          # Embedding: (9, 8)

# PyTorch の Linear(8, 9) のデフォルト初期化に合わせてスケール
# 範囲 [-1/√8, 1/√8] の一様分布 (= 範囲内のどの数も同じ確率で出るランダム)
# √8 ≒ 2.83 なので 1÷2.83 ≒ 0.35。つまり -0.35〜0.35 の範囲
bound = 1.0 / np.sqrt(8)
linear_weight = np.random.uniform(-bound, bound, (8, vocab))  # (8, 9)
# np.zeros(n): 長さ n のゼロベクトルを作る → 全部ゼロで初期化 (最初は偏りなしでスタート)
linear_bias = np.zeros(vocab)                                  # (9,)

params = emb_weight.size + linear_weight.size + linear_bias.size
print(f"パラメータ数: {params}\n")

# --- ハイパーパラメータ ---
lr = 1.0

# --- 学習 ---
for step in range(300):

    # === Forward pass (順伝播) = データを前に流して予測する処理 ===

    # Embedding lookup: 各入力文字のインデックスに対応する行を取り出す
    # → 文字番号をもとに、各文字の特徴ベクトル (8個の数値) を取り出す
    h = emb_weight[X]  # (13, 8)

    # Linear: dot(h, W) + b = Σ_k h[i,k] * W[k,j] + b[j]
    # (= h の各要素と W の各要素を掛けて全部足す + バイアスを足す)
    # → 各文字の特徴ベクトルと重みの掛け算+足し算で、次の文字の"なりやすさスコア"を計算
    logits = np.dot(h, linear_weight) + linear_bias  # (13, 9)
    # logits = まだ確率ではなく、大きい値ほどその文字になりやすいという生のスコア

    # === CrossEntropyLoss (softmax + negative log likelihood) ===
    # = 予測がどれだけハズレたかを1つの数値 (loss) にする

    # 各行の最大値 max_j(logits[i,j]) を引く (数値安定性のため)
    # → スコアが大きすぎると e^x の計算で数値が爆発するので、安全な範囲に調整
    max_val = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - max_val  # オーバーフロー防止

    # exp(x): 各要素に e^x を計算
    # e (≒2.718) をスコア回掛ける。例: e^2 ≒ 7.4。スコアが大きいほど結果もグンと大きくなる
    exp_logits = np.exp(logits_shifted)  # (13, 9)

    # 各行の合計 Σ_j exp_logits[i,j]
    # → 各文字の exp 値を全部足す。この後割り算して合計1にするための準備
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)  # (13, 1)

    # softmax: probs[i,j] = exp(logits[i,j]) / Σ_j exp(logits[i,j])
    # → exp ÷ 合計 で 0〜1 の確率に変換。全文字足すとちょうど 1 になる
    probs = exp_logits / sum_exp  # (13, 9)

    # 各サンプルの正解文字の確率を取り出して loss 計算
    # loss = -1/N * Σ_i log(probs[i, Y[i]])
    # → 正解文字の確率が低いほどペナルティが大きい。全サンプルの平均が最終的な loss
    # np.log = log (対数) = "e を何乗したらこの数になるか"
    #   確率1.0 なら log=0 (ペナルティなし)、確率が 0 に近づくと -∞ に向かう (ペナルティ大)
    loss = 0.0
    for i in range(N):
        loss = loss + (-np.log(probs[i, Y[i]]))
    loss = loss / N

    if step % 100 == 0:
        print(f"  step {step:3d}  loss={loss:.3f}")

    # === Backward pass (逆伝播) = 予測を正解に近づけるには、重みをどう直せばいいか計算する ===
    # 勾配 (gradient) = 坂道の傾き。loss を減らすにはどっちの方向に重みを動かせばいいかを示す値

    # softmax + cross-entropy の勾配: dL/d(logits) = probs - one_hot(Y)
    # one_hot = 正解の位置だけ 1、他は 0 のベクトル
    # つまり予測確率から正解位置だけ 1 を引く
    # 例: "も"が正解で確率 0.3 なら 0.3-1=-0.7 → "もっと確率を上げろ" という修正指示
    # これは数学的に導出できる有名な結果
    dlogits = probs.copy()  # (13, 9)
    for i in range(N):
        dlogits[i, Y[i]] = dlogits[i, Y[i]] - 1
    dlogits = dlogits / N

    # Linear の重みの勾配: dW = h^T @ dlogits = Σ_i h[i,k] * dlogits[i,j]
    # → 入力 (h) と修正指示 (dlogits) の掛け算で、重みの修正量を求める
    dlinear_weight = np.dot(h.T, dlogits)  # (8, 9)

    # Linear のバイアスの勾配: db = Σ_i dlogits[i,j]
    # → 修正指示を全サンプル分足し合わせたもの
    dlinear_bias = np.sum(dlogits, axis=0)  # (9,)

    # Embedding 出力の勾配: dh = dlogits @ W^T = Σ_j dlogits[i,j] * W[k,j]
    # → 修正指示を重みの逆方向に流して、Embedding にも "こう直して" と伝える (これが "逆"伝播)
    dh = np.dot(dlogits, linear_weight.T)  # (13, 8)

    # Embedding の重みの勾配: 同じ文字に複数の勾配が来るので足し合わせる
    # → 同じ文字 (例: "も") がテキスト中に何度も出てくるので、それぞれの修正指示を全部足し合わせる
    # np.zeros_like(x): x と同じ形のゼロ行列を作る
    demb_weight = np.zeros_like(emb_weight)  # (9, 8)
    for i in range(N):
        demb_weight[X[i]] = demb_weight[X[i]] + dh[i]

    # === SGD (確率的勾配降下法) 更新: w = w - lr * dw ===
    # = 坂道を下るように loss が減る方向に重みを少しずつ動かす方法
    # lr (学習率) = 1回の更新でどれだけ大きく動かすかの設定値
    emb_weight = emb_weight - lr * demb_weight
    linear_weight = linear_weight - lr * dlinear_weight
    linear_bias = linear_bias - lr * dlinear_bias

# --- 生成: 1文字ずつ予測を繰り返す ---
print("\n--- 生成 ---")
ch = "も"
result = ch
for _ in range(15):
    x_idx = to_i[ch]

    # Embedding lookup: 1文字分のベクトルを取り出す
    h = emb_weight[x_idx]  # (8,)

    # Linear: dot(h, W) + b
    logits = np.dot(h, linear_weight) + linear_bias  # (9,)

    # softmax でスコアを確率に変換 (学習時と同じ計算)
    logits_shifted = logits - np.max(logits)  # 最大値を引く (数値安定性のため)

    exp_logits = np.exp(logits_shifted)  # e^x を計算

    # softmax: p[j] = exp(logits[j]) / Σ_j exp(logits[j])
    probs = exp_logits / np.sum(exp_logits)

    # 確率に従ってサイコロを振るように1文字を選ぶ
    ch = to_c[np.random.choice(vocab, p=probs)]
    result = result + ch

print(result)
