"""
本当に最小のAI - たった1層のニューラルネットワーク (numpy版)

== そもそもAIって何をやってるの？ ==

AIがやってることは「次の文字を当てるゲーム」、ただそれだけ。
「も」と入力されたら「次は"も"かな？"た"かな？」と予測を返す。
ChatGPTのような大規模なものも、根っこの仕組みはこれと同じ。

== じゃあどうやって当てるの？ ==

プログラムの中に「数値がたくさん入った表」がある。
入力された文字とこの表の数値を使って計算すると、予測が1つ出てくる。

最初、表には適当な数値が入ってるから、予測は全然当たらない。
そこで「どのくらいハズレたか」を数値で測って、表の中の数値をちょっとだけ直す。
また予測する → またハズレる → また直す。
これを何百回も繰り返すと、表の数値がだんだん良くなって、予測が当たるようになる。

AIの世界では、この表の数値のことを「重み (weight)」と呼ぶ。
そして、重みを何度も直していく作業のことを「学習 (training)」と呼ぶ。
でも中身は「表の数値を書き換えてるだけ」。記憶とか知能とかは関係ない。

全体の流れ:
  ① 文字を数値に変換する (コンピュータは文字を直接扱えないから)
  ② 数値の表を使って「次の文字」を予測する
  ③ 予測がどれだけハズレたかを測る
  ④ ハズレを減らす方向に表の数値を直す
  ⑤ ②〜④を300回繰り返す
  ⑥ 良くなった表を使って文章を生成する

※ 各操作の専門用語(Embedding, softmax 等)はコード末尾にまとめてある。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.floating]
NDArrayInt = npt.NDArray[np.integer]


# =============================================
# ① 文字を数値に変換する
# =============================================
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


# =============================================
# ② 〜 ⑤ の本体 (予測・ハズレ測定・修正)
# =============================================
class TinyLM:
    def __init__(self, vocab_size: int, emb_dim: int = 8) -> None:
        # --- 数値の表を用意する ---
        # ここで作る3つの表が、このプログラムの全て。
        # 最初は適当な数値が入っていて、②〜④を繰り返すことで良い値になっていく。

        # 表1: 文字ごとの特徴リスト (9文字 × 8個の数値)
        # 文字番号を使ってこの表を引くと、その文字の「特徴」が8個の数値として出てくる。
        # 最初は適当な数値だから「特徴」と呼ぶのもおこがましいけど、
        # 何度も数値を直していくうちに、似た使われ方をする文字は似た数値になっていく。
        self.emb_weight: NDArrayFloat = np.random.randn(vocab_size, emb_dim)          # (9, 8)

        # 表2と表3: 特徴から「次の文字のスコア」を計算するための表
        # 表1から取り出した8個の数値を、表2・表3と計算することで、
        # 「次にどの文字が来そうか」のスコアを全文字分(9個)出す。
        # 中身はかけ算と足し算だけ。
        bound = 1.0 / np.sqrt(emb_dim)
        self.linear_weight: NDArrayFloat = np.random.uniform(-bound, bound, (emb_dim, vocab_size))  # 表2: (8, 9)
        self.linear_bias: NDArrayFloat = np.zeros(vocab_size)                                        # 表3: (9,)

    @property
    def param_count(self) -> int:
        return self.emb_weight.size + self.linear_weight.size + self.linear_bias.size

    # =============================================
    # ② 数値の表を使って「次の文字」を予測する
    # =============================================
    # データを前に流して予測を出す処理。
    # 入力文字 → 表を引く → 計算 → 予測、という一方通行の流れ。
    def forward(self, X: NDArrayInt) -> tuple[NDArrayFloat, NDArrayFloat]:

        # --- 表1を引く ---
        # 入力文字の番号を使って、表1からその文字の行を取り出す。
        # 例: 「も」= 番号4 → 表1の4行目 → [0.3, -1.2, 0.8, ...] (8個の数値)
        h: NDArrayFloat = self.emb_weight[X]  # (13, 8)

        # --- 表2, 表3で計算する ---
        # 取り出した8個の数値と、表2の数値を掛けて足し合わせ、さらに表3を足す。
        # すると「次にどの文字が来そうか」のスコアが、全9文字分出てくる。
        # 例: う=1.2, か=-0.5, た=3.1, ... → 「た」のスコアが一番高い → 「た」が来そう
        #
        # やってることは (8個の数値) × (8×9の表) + (9個の数値) = 9個のスコア
        # 全部ただのかけ算と足し算。
        logits: NDArrayFloat = np.dot(h, self.linear_weight) + self.linear_bias  # (13, 9)

        # --- スコアをパーセントに変換する ---
        # ここまでで各文字のスコアは出た。でもスコアのままだと困る。
        # 例: た=3.1, も=2.8 で「どっちが何%くらいありそう？」がわからない。
        # マイナスの値もあるし、スコアの合計も意味がない。
        #
        # そこで全部「パーセント(確率)」に変換する。足すとちょうど100%になるように。
        # 例: う=5%, か=2%, た=40%, も=30%, ... (合計100%)
        # こうすれば「たが来る確率は40%」と言えるようになる。
        #
        # 手順:
        #   1. まず全スコアからいちばん大きい値を引く
        #      (この後の計算で数値が爆発するのを防ぐため。結果は変わらない)
        max_val: NDArrayFloat = np.max(logits, axis=1, keepdims=True)
        logits_shifted: NDArrayFloat = logits - max_val

        #   2. 各スコアに対して「eのスコア乗」を計算する
        #      e ≒ 2.718 で、スコアが大きいほど結果がドンと大きくなる。
        #      これによってスコアの差が強調される。
        #      (たとえばスコア3と1の差は2だけど、e^3=20 と e^1=2.7 で7倍以上の差になる)
        exp_logits: NDArrayFloat = np.exp(logits_shifted)  # (13, 9)

        #   3. 全部足して、各値をその合計で割る → 合計がちょうど1(=100%)になる
        sum_exp: NDArrayFloat = np.sum(exp_logits, axis=1, keepdims=True)  # (13, 1)
        probs: NDArrayFloat = exp_logits / sum_exp  # (13, 9)
        # これで probs が「各文字が次に来る確率」の予測になった。

        return probs, h

    # =============================================
    # ③ 予測がどれだけハズレたかを測る
    # =============================================
    # 予測が出たら、次は「どのくらい間違えたか」を1つの数値にしたい。
    # この数値が大きいほどハズレがひどい、小さいほど良い予測ということ。
    #
    # 考え方はシンプル:
    #   正解の文字に何%の確率を割り振れたかを見る。
    #   正解が「た」で、「た」に80%を振れていたら → まあまあ良い → ハズレは小さい
    #   正解が「た」で、「た」に2%しか振れてなかったら → ひどい → ハズレは大きい
    #
    # 具体的にはlog(正解の確率)にマイナスをつけた値を使う。
    #   確率80% → -log(0.8) ≒ 0.22 (小さい = 良い)
    #   確率2%  → -log(0.02) ≒ 3.9  (大きい = ハズレ)
    #   確率100%なら → -log(1.0) = 0  (完璧、ハズレなし)
    # なぜlogを使うかというと、確率が低いときにドカンとペナルティが跳ね上がるから。
    # 50%→10%の悪化より、10%→1%の悪化のほうが深刻で、logはそれを自然に反映してくれる。
    def loss(self, probs: NDArrayFloat, Y: NDArrayInt) -> float:
        N = len(Y)
        loss = 0.0
        for i in range(N):
            loss = loss + (-np.log(probs[i, Y[i]]))
        return loss / N  # 全サンプルの平均をとる

    # =============================================
    # ④ ハズレを減らす方向に表の数値を直す
    # =============================================
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

        # --- まず、スコアに対する修正指示を求める ---
        # ここの計算は「予測確率 - 正解」というシンプルな形になる。
        # 例: 「た」が正解で、予測が「た=40%, も=30%, ...」だったら、
        #   「た」のところ: 0.4 - 1 = -0.6 → 「もっと確率を上げろ」
        #   「も」のところ: 0.3 - 0 = +0.3 → 「確率を下げろ」
        dlogits: NDArrayFloat = probs.copy()  # (13, 9)
        for i in range(N):
            dlogits[i, Y[i]] = dlogits[i, Y[i]] - 1
        dlogits = dlogits / N

        # --- 表2, 表3への修正指示を求める ---
        # 上で求めた修正指示を元に、表2と表3をどう直すか計算する。
        dlinear_weight: NDArrayFloat = np.dot(h.T, dlogits)    # 表2への修正指示 (8, 9)
        dlinear_bias: NDArrayFloat = np.sum(dlogits, axis=0)   # 表3への修正指示 (9,)

        # --- 表1への修正指示を求める ---
        # 修正指示を逆方向に流して、表1にも「こう直して」と伝える。
        # 予測の最後で出たハズレを、逆向きにたどって表1まで伝えていく。
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

    # =============================================
    # ⑥ 良くなった表を使って文章を生成する
    # =============================================
    # 300回も数値を直した表は、もうだいぶ良い予測ができるようになっている。
    # あとは1文字ずつ「予測→その文字を次の入力にする→また予測→...」を繰り返すだけ。
    def generate(self, start_char: str, length: int, vocab: Vocab) -> str:
        ch = start_char
        result = ch
        for _ in range(length):
            x_idx = vocab.to_i[ch]

            # ②と同じ処理を1文字ぶんだけやる
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


# =============================================
# ここから実行
# =============================================

# === ① 文字を数値に変換する ===
text: str = "ももたろうはももからうまれた"

vocab: Vocab = Vocab(text)
print(f"文字数: {len(text)}  語彙: {vocab.size}種 {vocab.chars}")

# 入力と正解のペアを作る: 「1文字目→2文字目」「2文字目→3文字目」...
# 例: 「ももたろう...」→ 入力「ももたろ...」、正解「もたろう...」
X: NDArrayInt = vocab.encode(text[:-1])  # 入力: 先頭から最後の1つ手前まで
Y: NDArrayInt = vocab.encode(text[1:])   # 正解: 2文字目から最後まで

model: TinyLM = TinyLM(vocab.size)
print(f"パラメータ数: {model.param_count}\n")

# === ⑤ ②〜④を300回繰り返す ===
lr: float = 1.0  # 学習率: 1回の修正でどれだけ大きく動かすか

for step in range(300):
    probs: NDArrayFloat
    h: NDArrayFloat
    probs, h = model.forward(X)       # ② 予測する
    loss: float = model.loss(probs, Y)  # ③ ハズレを測る

    if step % 100 == 0:
        print(f"  step {step:3d}  loss={loss:.3f}")

    model.backward(probs, Y, h, X, lr)  # ④ 表の数値を直す

# === ⑥ 生成する ===
print("\n--- 生成 ---")
print(model.generate("も", 15, vocab))


# =============================================
# 専門用語との対応
# =============================================
# ここまで読めば、もう中身はわかっているはず。
# あとはそれぞれの操作に業界でどんな名前がついているかだけ。
#
#   このコードでの説明              → 専門用語
#   ─────────────────────────────────────────────
#   表の数値                        → 重み (weight) / パラメータ (parameter)
#   表の数値を何度も直していく作業  → 学習 (training)
#   文字番号で表1を引いて取り出す   → Embedding (埋め込み)
#   かけ算と足し算でスコアを出す    → Linear (線形変換)
#   スコアをパーセントに変換する    → softmax (ソフトマックス)
#   ハズレを1つの数値にしたもの     → Cross Entropy Loss (交差エントロピー損失)
#   各数値への修正指示              → 勾配 (gradient)
#   修正指示を後ろから前に伝える    → 逆伝播 (Backpropagation)
#   修正指示で数値を直すやり方      → SGD (確率的勾配降下法)
#   データを前に流して予測を出す    → 順伝播 (Forward pass)
