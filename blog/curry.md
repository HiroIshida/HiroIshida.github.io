---
title: 美味しいカレーの作り方
date: 2026-10-12
author: Hirokazu Ishida
description: 玉ねぎを炒め、スパイスを加え、なぜか目的関数を定義する。数式と調理工程図で考える、美味しいカレーの作り方。
---

記念すべき最初の記事なので、壮大な話をしようと思った。けれど、お腹が空いたのでカレーを作ることにした。せっかくなので、鍋の中で起きることを少しだけ研究っぽく眺めてみる。

以下の数式は料理を大げさに説明するための遊びで、実測に基づくモデルではありません。末尾の参考文献もすべて架空です。カレーの手順は普通に使えます。
{: .aside}

## まず、目的関数を定義する {#objective}

「美味しい」では仕様が曖昧すぎる。そこで、香りを \(A\)、うま味を \(U\)、好みの塩加減への近さを \(S\)、焦げによる苦味を \(B\) とし、それぞれを 0〜1 の気分で評価する。今回の目的関数は次の通りである。<a aria-label="架空の参考文献1" class="citation" href="#ref-1">[1]</a>

<figure class="equation">
  <div class="math-scroll" tabindex="0" role="region" aria-label="式1：カレーの美味しさの目的関数">
    \[
      \begin{aligned}
        \boldsymbol{x}^{\star} &amp;= \arg\max_{\boldsymbol{x}} J(\boldsymbol{x}) \\
        J(\boldsymbol{x}) &amp;= 2A + U + S - 5B
      \end{aligned}
    \]
  </div>
  <figcaption>(1)</figcaption>
</figure>

操作する変数 \(\boldsymbol{x}\) は、火加減、加熱時間、塩の量など。係数に根拠はない。ただ、「焦がしたときの悲しさ」を重く評価している。香りを少し増やすために鍋底を炭にしてはいけない。

## 実験装置と材料 {#ingredients}

必要な装置は、厚手の鍋、木べら、包丁、まな板。GPUは要らない。ご飯は先に炊いておく。これは計算と通信のオーバーラップに相当する。

表1. 2人分の入力パラメータ（調理時間の目安：約40分）
{: .aside}

| 材料 | 分量 | 担当 |
| --- | --- | --- |
| 玉ねぎ | 1個 | 甘みと土台 |
| 鶏もも肉 | 250 g | うま味と満足感 |
| カットトマト | 200 g | 酸味とうま味 |
| にんにく・生姜 | 各1かけ | 香りの立ち上がり |
| カレー粉 | 大さじ1 | 主役の香り |
| サラダ油 | 大さじ1 | 炒めるための油 |
| 水 | 150 mLから調整 | 煮込みと濃度調整 |
| 塩 | 小さじ1/2を目安に調整 | 味の輪郭 |
| 無糖ヨーグルト | 大さじ2 | 仕上げのまろやかさ |
| ご飯 | 2人分 | 最終出力の受け皿 |

カレー粉は、塩の入っていないスパイスミックスを想定。市販のカレールウを使う場合は、箱に書かれた分量と手順を優先する。先行研究を尊重したい。
{: .aside}

## 提案手法：炒めて、煮て、味見する {#method}

全体の構成を図1に示す。重要なのは、最後の味見から前の工程へ戻るループである。一度で正解を出そうとせず、少しずつ近づける。

<figure class="diagram" aria-labelledby="diagram-caption">
  <div class="diagram-panel">
    <div class="flow">
      <div class="flow-node">
        <strong>炒める</strong>
        <p class="node-input">玉ねぎ・油<br>にんにく・生姜・カレー粉</p>
        <p class="node-output">香りと甘みの土台</p>
      </div>
      <span class="flow-arrow" aria-hidden="true"><span class="arrow-desktop">→</span><span class="arrow-mobile">↓</span></span>
      <div class="flow-node">
        <strong>煮込む</strong>
        <p class="node-input">鶏肉・トマト・水<br>弱火で、ふつふつと</p>
        <p class="node-output">具材に火を通し、味をまとめる</p>
      </div>
      <span class="flow-arrow" aria-hidden="true"><span class="arrow-desktop">→</span><span class="arrow-mobile">↓</span></span>
      <div class="flow-node">
        <strong>味見する</strong>
        <p class="node-input">塩・ヨーグルト<br>最後は自分の舌で評価</p>
        <p class="node-output">ちょうどよければ、ご飯へ</p>
      </div>
    </div>
    <div class="feedback">↺ まだ薄い？ 塩・水・煮込み時間を調整して、もう一度。</div>
  </div>
  <figcaption id="diagram-caption">図1. カレー生成システムの構成。味見によるフィードバックを持つ。評価器は人間なので、お腹の空き具合によって出力が変動する。</figcaption>
</figure>

### 1. 玉ねぎを、焦らず炒める

玉ねぎは薄切り、にんにくと生姜はみじん切り、鶏肉はひと口大にする。鍋に油を入れ、玉ねぎを中火で10〜15分ほど炒める。薄いきつね色になれば十分。焦げ付きそうなら火を弱め、水を少量足して鍋底をこそげる。飴色という単語に人生を支配されないこと。

### 2. 香りを立てて、トマトを加える

にんにくと生姜を加えて1分ほど炒める。弱火にしてカレー粉を加え、20〜30秒ほど混ぜたら、トマトを投入する。中火に戻し、へらで混ぜながら3〜5分ほど煮詰める。スパイスを入れてからメールを確認するのはやめよう。焦げのペナルティ係数は5である。<a aria-label="架空の参考文献2" class="citation" href="#ref-2">[2]</a>

### 3. 鶏肉を加えて、弱火で煮込む

鶏肉を加えて全体にからめ、水150 mLと塩の半量を加える。煮立ったら弱火にし、ふたを少しずらして15〜20分ほど、鶏肉の中心まで十分に火が通るまで煮る。ときどき混ぜ、水分が足りなければ少量ずつ足す。ここで初めて、鍋の前で何もしない時間が生まれる。

### 4. 味を整えて、ご飯にかける

火を止め、ヨーグルトを混ぜる。少量を取り、少し冷ましてから味見する。残りの塩は一度に入れず、必要な分だけひとつまみずつ。濃ければ水でのばし、薄ければ弱火で少し煮詰める。ひと手間ごとに味見して、ちょうどよくなったところでご飯にかける。

## 味見はフィードバック制御である {#feedback}

現在の塩の量を \(s_k\)、味の物足りなさを \(e_k\) とすると、塩の追加は次の更新則として書ける。<a aria-label="架空の参考文献3" class="citation" href="#ref-3">[3]</a>

<figure class="equation">
  <div class="math-scroll" tabindex="0" role="region" aria-label="式2：味見に基づく塩の調整">
    \[
      s_{k+1} = s_k + \eta\,\max(0, e_k)
    \]
  </div>
  <figcaption>(2)</figcaption>
</figure>

\(\eta\) は一度に足す量を決める係数である。要するに、**塩は少しずつ入れる**。大きな更新幅は素早い収束をもたらすかもしれないが、塩辛さへの発散も招く。しかも、一度入れた塩を鍋から取り出す操作は実装されていない。

美味しくなったら終了する。最適解であることの証明は要らない。温かいうちに食べるほうが大切だ。

玉ねぎを炒める。スパイスを焦がさない。塩は少しずつ。数式を全部読み飛ばしても、カレーは美味しく作れる。

## 参考文献 {#references}

以下は記事の体裁を楽しむためのダミーです。著者・論文・掲載誌はいずれも架空で、実在の研究を引用したものではありません。
{: .aside}

<ol class="references">
  <li id="ref-1"><cite>玉葱 太郎・香辛 花子, 「家庭内カレー最適化における空腹バイアスの影響」</cite>, <i>日本夕飯学会誌</i>, vol. 1, no. 1, pp. 1–8, 2026.</li>
  <li id="ref-2"><cite>C. Cumin and T. Turmeric, “Attention Is All You Need While Frying Spices,”</cite> <i>Proceedings of the International Conference on Computational Curry</i>, pp. 42–49, 2025.</li>
  <li id="ref-3"><cite>鍋底 まもる, 「ひとつまみずつ：不可逆な調味操作のための保守的フィードバック制御」</cite>, <i>Transactions on Taste and Control</i>, vol. 3, no. 2, pp. 15–22, 2026.</li>
</ol>
