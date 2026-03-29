"""
本当に最小のAI - たった1層のニューラルネットワーク (numpy版)

== そもそもAIって何をやってるの？ ==

AIがやってることは「次の文字を当てるゲーム」。
「も」と入力されたら「次は"も"かな？"た"かな？」と予測を返す。
ChatGPTのような大規模なものも、根っこの仕組みはこれと同じ。

== このプログラムがやること ==

  1. 予測: 文字を入れたら、表の数値を使って「次の文字」の予測が出る
  2. 学習: 予測して → ハズレを測って → 表を直す、を300回繰り返す
  3. 生成: 学習した表を使って、予測を繰り返して文章を作る

※ 各操作の専門用語(Embedding, softmax 等)はコード末尾にまとめてある。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.floating]
NDArrayInt = npt.NDArray[np.integer]


# ==============================================================================
# 予測の準備: 文字を数値に変換する
# ==============================================================================
# コンピュータは「も」「た」みたいな文字をそのまま計算に使えない。
# だから最初に、文字に番号を振る。
# 例: う=0, か=1, た=2, は=3, も=4, ら=5, ろ=6, ま=7, れ=8
# これで「もも」は [4, 4] という数値の列になって、計算に使える。
class Vocab:
    def __init__(self, text: str) -> None:
        self.chars: list[str] = sorted(set(text))
        self.size: int = len(self.chars)  # 文字の種類数
        self.to_i: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.to_c: dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> NDArrayInt:
        return np.array([self.to_i[c] for c in text])


class TinyLM:
    # ==========================================================================
    # 予測に使う3つの表を用意する
    # ==========================================================================
    # じゃあどうやって次の文字を当てるのか？
    # プログラムの中に「数値がたくさん入った表」がある。
    # 入力された文字とこの表の数値を使って計算すると、予測が1つ出てくる。
    # 予測には3つの「数値の表」を使う。
    # この3つがこのプログラムの全て。合計153個の数値。
    # 最初は適当な数値が入っていて、学習を繰り返すことで良い値になっていく。
    def __init__(self, vocab_size: int, emb_dim: int = 8) -> None:

        # 文字の変換表 (9文字 × 8個の数値 = 72個)
        # 文字番号を渡すと、その文字を8個の数値に変換して返してくれる表。
        # 辞書みたいなもの。最初は適当な数値だけど、
        # 何度も数値を直していくうちに、似た使われ方をする文字は似た数値になっていく。
        self.emb_weight: NDArrayFloat = np.random.randn(vocab_size, emb_dim)          # (9, 8)

        # 予測の点数表 (8 × 9 = 72個)
        # 変換表から出てきた8個の数値と掛け合わせて、
        # 次にどの文字が来そうかの「点数」を出すための表。
        # 点数表の中の数値が大きいところほど結果に強く影響する。
        # 学習を繰り返すと、正解の文字に高い点数がつくように数値が調整されていく。
        bound = 1.0 / np.sqrt(emb_dim)
        self.linear_weight: NDArrayFloat = np.random.uniform(-bound, bound, (emb_dim, vocab_size))  # (8, 9)

        # 予測の調整値 (9個)
        # 点数に足す微調整の数値。
        # 掛け算だけだと「この文字はそもそも来やすい/来にくい」という
        # 全体的な底上げ・底下げができない。
        # テストで例えると、点数表が「配点×回答」の素点なら、
        # 調整値は「この科目は難しかったから全員に+5点」のような加点。
        self.linear_bias: NDArrayFloat = np.zeros(vocab_size)                                        # (9,)

    @property
    def param_count(self) -> int:
        return self.emb_weight.size + self.linear_weight.size + self.linear_bias.size

    # ==========================================================================
    # 予測: 文字を入れたら予測が出る
    # ==========================================================================
    # 入力文字 → 変換表を引く → 点数表・調整値で計算 → パーセントに変換 → 予測完了
    # という一方通行の流れ。
    def forward(self, X: NDArrayInt) -> tuple[NDArrayFloat, NDArrayFloat]:

        # --- 文字の変換表を引く ---
        # 入力文字の番号を使って、変換表からその文字の行を取り出す。
        # 例: 「も」= 番号4 → 変換表の4行目 → [0.3, -1.2, 0.8, ...] (8個の数値)
        h: NDArrayFloat = self.emb_weight[X]  # (13, 8)

        # --- 点数表・調整値で計算する ---
        # 変換表から取り出した8個の数値と、点数表の数値を1つずつ掛けて全部足す。
        # さらに調整値を足す。
        # これで「次にどの文字が来そうか」の点数が、全9文字分出てくる。
        # 例: う=-0.5, か=0.3, た=3.1, ... → 「た」の点数が一番高い → 「た」が来そう
        logits: NDArrayFloat = np.dot(h, self.linear_weight) + self.linear_bias  # (13, 9)

        # --- 点数をパーセントに変換する ---
        # 点数のままだと「た=3.1, も=2.8」で何%ありそうか比べにくいし、
        # マイナスの値もある。そこで、どんな数値でもパーセントに変換できる
        # 数学の公式を使う (np.exp → 合計で割る)。
        max_val: NDArrayFloat = np.max(logits, axis=1, keepdims=True)
        logits_shifted: NDArrayFloat = logits - max_val  # 数値が大きすぎて壊れるのを防ぐ前処理
        exp_logits: NDArrayFloat = np.exp(logits_shifted)  # 全部正の数に変換 (13, 9)
        sum_exp: NDArrayFloat = np.sum(exp_logits, axis=1, keepdims=True)  # (13, 1)
        probs: NDArrayFloat = exp_logits / sum_exp  # 合計で割って100%にする (13, 9)

        return probs, h

    # ==========================================================================
    # 学習①: ハズレを測る
    # ==========================================================================
    # 予測の仕組みはわかった。でも最初は表の数値が適当だから、予測は全然当たらない。
    # ここからが本番で、予測 → ハズレ測定 → 修正 を繰り返して、表の数値を良くしていく。
    # AIの世界では、この表の数値のことを「重み (weight)」、
    # 重みを何度も直していく作業のことを「学習 (training)」と呼ぶ。
    # でも中身は「表の数値を書き換えてるだけ」。
    #
    # まずはハズレの測り方から。
    # 予測が出たら「どのくらい間違えたか」を1つの数値にしたい。
    # この数値が大きいほどハズレがひどい、小さいほど良い予測ということ。
    #
    # 考え方はシンプル:
    #   正解の文字に何%の確率を割り振れたかを見る。
    #   正解が「も」で、「も」に60%を振れていたら → 良い → ハズレは小さい
    #   正解が「も」で、「も」に2%しか振れてなかったら → ひどい → ハズレは大きい
    #
    # 正解の確率が低いほどハズレの度合いが大きくなる、という計算をしたい。
    # これにぴったりな数学の関数 (log) がある。
    # logは確率を入れると0かマイナスの値を返す関数で、確率が低いほど大きなマイナスになる。
    #   log(0.8) = -0.22,  log(0.1) = -2.30,  log(0.02) = -3.91
    # ハズレは「大きいほど悪い」にしたいのに結果がマイナスなので、
    # マイナスをつけて符号をひっくり返す → -np.log(確率)。
    #   -log(0.8) = 0.22 (ハズレほぼなし),  -log(0.02) = 3.91 (ハズレひどい)
    def loss(self, probs: NDArrayFloat, Y: NDArrayInt) -> float:
        N = len(Y)
        loss = 0.0
        for i in range(N):
            loss = loss + (-np.log(probs[i, Y[i]]))
        return loss / N  # 全サンプルの平均をとる

    # ==========================================================================
    # 学習②: 表の数値を直す
    # ==========================================================================
    # ハズレを測ったら、次は
    # 「表のどの数値を、どっちの方向に、どれだけ直せばいいか」を計算する。
    #
    # イメージ: 山の中で目隠しされている。一番低い谷に行きたい(= ハズレを最小にしたい)。
    # 足元の傾きを感じて「こっちが下り坂だ」とわかったら、そっちに一歩進む。
    # また傾きを感じて、また一歩。これを繰り返して谷を目指す。
    # 「足元の傾き」が修正指示で、「一歩進む」が表の数値を直す作業。
    #
    # 実際に表の数値を直す式はこれだけ:
    #   新しい数値 = 今の数値 - 学習率 × 修正指示
    # 「学習率」は一歩の大きさ。大きすぎると谷を飛び越える。小さすぎると進みが遅い。
    def backward(self, probs: NDArrayFloat, Y: NDArrayInt, h: NDArrayFloat, X: NDArrayInt, lr: float) -> None:
        N = len(Y)

        # --- まず、点数に対する修正指示を求める ---
        # 計算は「予測確率 - 正解」。
        # 正解の文字は「確率が足りない分だけ上げろ」という指示になり、
        # 不正解の文字は「今の確率ぶんだけ下げろ」という指示になる。
        # 例: 「も」が正解で、予測が「も=45%, た=20%, ...」だったら、
        #   「も」のところ: 0.45 - 1 = -0.55 → 「あと55%ぶん上げろ」
        #   「た」のところ: 0.20 - 0 = +0.20 → 「20%ぶん下げろ」
        dlogits: NDArrayFloat = probs.copy()  # (13, 9)
        for i in range(N):
            dlogits[i, Y[i]] = dlogits[i, Y[i]] - 1
        dlogits = dlogits / N

        # --- 点数表・調整値への修正指示を求める ---
        dlinear_weight: NDArrayFloat = np.dot(h.T, dlogits)    # 点数表への修正指示 (8, 9)
        dlinear_bias: NDArrayFloat = np.sum(dlogits, axis=0)   # 調整値への修正指示 (9,)

        # --- 変換表への修正指示を求める ---
        # 修正指示を逆方向にたどって、変換表にも「こう直して」と伝える。
        dh: NDArrayFloat = np.dot(dlogits, self.linear_weight.T)  # (13, 8)

        # 同じ文字が複数回出てくる場合(例:「も」が3回)、それぞれの修正指示を足し合わせる。
        demb_weight: NDArrayFloat = np.zeros_like(self.emb_weight)  # (9, 8)
        for i in range(N):
            demb_weight[X[i]] = demb_weight[X[i]] + dh[i]

        # --- 実際に表の数値を直す ---
        # 修正指示の方向に、学習率ぶんだけ数値を動かす。
        self.emb_weight = self.emb_weight - lr * demb_weight
        self.linear_weight = self.linear_weight - lr * dlinear_weight
        self.linear_bias = self.linear_bias - lr * dlinear_bias

    # ==========================================================================
    # 生成: 学習した表で文章を作る
    # ==========================================================================
    # 300回も数値を直した表は、もうだいぶ良い予測ができるようになっている。
    # あとは予測を繰り返すだけ:
    #   1. 「も」を入力 → 予測が出る → 確率に従ってサイコロを振り「も」を選ぶ
    #   2. 「も」を入力 → 予測が出る → サイコロを振り「た」を選ぶ
    #   3. 「た」を入力 → 予測が出る → サイコロを振り「ろ」を選ぶ
    #   4. ...こうして「ももたろうは...」のような文章ができあがる
    def generate(self, start_char: str, length: int, vocab: Vocab) -> str:
        ch = start_char
        result = ch
        for _ in range(length):
            x_idx = vocab.to_i[ch]

            # 「予測」と同じ処理を1文字ぶんだけやる
            h: NDArrayFloat = self.emb_weight[x_idx]
            logits: NDArrayFloat = np.dot(h, self.linear_weight) + self.linear_bias
            logits_shifted: NDArrayFloat = logits - np.max(logits)
            exp_logits: NDArrayFloat = np.exp(logits_shifted)
            probs: NDArrayFloat = exp_logits / np.sum(exp_logits)

            # 予測された確率に従って、サイコロを振るように1文字を選ぶ。
            # 40%の文字は40%の確率で選ばれ、2%の文字は2%の確率で選ばれる。
            ch = vocab.to_c[np.random.choice(vocab.size, p=probs)]
            result = result + ch

        return result


# ==============================================================================
# ここから実行
# ==============================================================================

# === 予測の準備: 文字を数値に変換する ===
text: str = "ももたろうはももからうまれた"

vocab: Vocab = Vocab(text)
print(f"文字数: {len(text)}  語彙: {vocab.size}種 {vocab.chars}")

# 入力と正解のペアを作る: 「1文字目→2文字目」「2文字目→3文字目」...
# 例: 「ももたろう...」→ 入力「ももたろ...」、正解「もたろう...」
X: NDArrayInt = vocab.encode(text[:-1])  # 入力: 先頭から最後の1つ手前まで
Y: NDArrayInt = vocab.encode(text[1:])   # 正解: 2文字目から最後まで

model: TinyLM = TinyLM(vocab.size)
print(f"パラメータ数: {model.param_count}\n")

# === 学習: 予測→ハズレ測定→修正 を300回繰り返す ===
lr: float = 1.0  # 学習率: 1回の修正でどれだけ大きく動かすか

for step in range(300):
    probs: NDArrayFloat
    h: NDArrayFloat
    probs, h = model.forward(X)       # 予測する
    loss: float = model.loss(probs, Y)  # ハズレを測る

    if step % 100 == 0:
        print(f"  step {step:3d}  loss={loss:.3f}")

    model.backward(probs, Y, h, X, lr)  # 表の数値を直す

# === 生成: 学習した表で文章を作る ===
print("\n--- 生成 ---")
print(model.generate("も", 15, vocab))


# ==============================================================================
# 専門用語との対応
# ==============================================================================
# ここまで読めば、もう中身はわかっているはず。
# あとはそれぞれの操作に業界でどんな名前がついているかだけ。
#
#   このコードでの説明              → 専門用語
#   ─────────────────────────────────────────────
#   表の数値                        → 重み (weight) / パラメータ (parameter)
#   表の数値を何度も直していく作業  → 学習 (training)
#   文字番号で変換表を引いて取り出す → Embedding (埋め込み)
#   掛け算と足し算で点数を出す      → Linear (線形変換)
#   点数をパーセントに変換する      → softmax (ソフトマックス)
#   ハズレを1つの数値にしたもの     → Cross Entropy Loss (交差エントロピー損失)
#   各数値への修正指示              → 勾配 (gradient)
#   修正指示を後ろから前に伝える    → 逆伝播 (Backpropagation)
#   修正指示で数値を直すやり方      → SGD (確率的勾配降下法)
#   データを前に流して予測を出す    → 順伝播 (Forward pass)
