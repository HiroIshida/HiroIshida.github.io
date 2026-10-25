---
title: [WIP] LLMの層プルーニング：ShortGPTからLLM-Streamlineまで
date: 2026-10-25
author: Hirokazu Ishida
description: ShortGPT、The Unreasonable Ineffectiveness of the Deeper Layers、LLM-Streamlineを、層と区間の選択、性能回復、データ依存性とタスクごとの劣化から整理する。
---

学習済みLLMの層を丸ごと取り除くと、どこまで能力を保てるのでしょうか。層の入出力が似ているなら削れそうですが、一層ずつ似ていることと、複数層をまとめて飛ばしてもよいことは同じではありません。また、知識を問う選択問題が解けることと、文章生成や数学の推論ができることも、層を削った後には別々に評価する必要があります。本稿では、何を削除の単位にするか、削除後のずれをどう補うか、どの能力が実際に残るかを、3本の論文から整理します。

順序はShortGPT、The Unreasonable Ineffectiveness of the Deeper Layers、LLM-Streamlineです。初稿はそれぞれ2024年3月6日、26日、28日で、2本目はShortGPTを、3本目は前の2本を引用しています。「一層ごとの重要度による削除 → 連続区間の選択と追加学習 → 区間を小さなネットワークで置き換える」という説明上のつながりに沿って読みます。ただし同時期の研究であり、後の改訂ではShortGPTも3本目を引用しているため、単純な一方向の派生関係ではありません。基本の参照版は順にarXiv v3、v2、v5とし、ShortGPTのデータ依存性についてはACL 2025掲載版の追加実験も扱います。<a class="citation" href="#ref-1">[1]</a><a class="citation" href="#ref-2">[2]</a><a class="citation" href="#ref-3">[3]</a><a class="citation" href="#ref-4">[4]</a>

## 1. ShortGPT：隠れ状態をあまり変えない層を削る

### 1.1 Block Influenceで各層を順位付けする

ShortGPTの基本手法は、重要度の測定用に少量のテキスト（calibration data）をモデルに通し、各層の入出力のcosine similarityを測って、変化の小さい層を削るというものです。ここでの一層はattentionとfeed-forward network（FFN）を含むTransformer blockです。層番号を \(\ell=0,\ldots,L-1\)、その層への入力を \(H_\ell\in\mathbb{R}^{T\times d}\) とします。\(T\) は系列長、\(d\) は隠れ状態の次元です。残差接続をまとめて \(H_{\ell+1}=H_\ell+F_\ell(H_\ell)\) と書けば、層を削る操作はこの変換を恒等写像に置き換えることに対応します。元の入出力が近い場所なら、その置き換えで後続の層に渡す表現のずれも小さそうだ、というのが出発点です。<a class="citation" href="#ref-1">[1]</a>

入力系列 \(x\) のtoken位置 \(t\) における隠れ状態を \(h_{\ell,t}(x)\in\mathbb{R}^{d}\) とすると、層の重要度を表すBlock Influence（BI）は次式です。

<div class="math-scroll">
\[
\mathrm{BI}_\ell
=1-\mathbb{E}_{x,t}\left[
\frac{h_{\ell,t}(x)^\top h_{\ell+1,t}(x)}
{\|h_{\ell,t}(x)\|_2\,\|h_{\ell+1,t}(x)\|_2}
\right].
\]
</div>

期待値はcalibration dataとtoken位置についての平均です。BIが小さいほど入出力の向きが近く、削除候補になります。元モデルで測ったBIを昇順に並べ、予算に応じた数の層を削るため、選ばれる層が連続している必要はありません。基本手法では重みの再学習も行いません。cosine similarityはベクトルの大きさを無視するので、これは出力誤差そのものを最小化する指標ではなく、層の重要度を近似する指標です。また、一層を削ると後続の入力も変わるため、元モデル上で測った個別スコアから、複数層を同時に削った損失が厳密に決まるわけではありません。<a class="citation" href="#ref-1">[1]</a>

### 1.2 知識問題を保てても、生成性能まで保てるとは限らない

arXiv v3のTable 2では、PG19でBIを測り、約4分の1のパラメータを削減しています。以下は同表からの抜粋で、矢印は元モデルからShortGPTへの変化です。13タスク平均には正解率以外の指標も含まれるため、単一タスクの正解率とは区別します。<a class="citation" href="#ref-1">[1]</a>

<div class="math-scroll" markdown="1">

| モデル | パラメータ削減率 | MMLU正解率（%） | 13タスク平均 |
| --- | --- | --- | --- |
| Llama2-7B | 27.1% | 45.39 → 43.96 | 47.78 → 41.24 |
| Llama2-13B | 24.6% | 55.00 → 54.69 | 55.14 → 50.53 |

</div>

Llama2-13Bでは40層中10層を削っても、MMLUの低下は0.31ポイントです。同じ表のLaCoは平均47.62、ShortGPTは50.53で、同じ24.6%削減でも単純な層削除が良い結果を出しています。一方、7Bの要約タスクXSumのスコアは19.40から0.67へ落ちています。「約9割の性能を維持」という平均だけでは、この差は見えません。圧縮による損失が比較手法より小さいことに加え、その損失がタスクに強く依存することが示されています。後のACL掲載版で追加された生成向けの手法ShortGPT-genについては、Appendix Bで補足します。<a class="citation" href="#ref-1">[1]</a><a class="citation" href="#ref-4">[4]</a>

深い層なら一様に不要というわけでもありません。Table 1のLlama2-7Bでは、最終層を丸ごと削るとPG19のperplexity（PPL）は7.60から13.37に増えます。最終層のattentionだけを削る場合は7.65ですが、FFNだけを削ると12.35です。PPLは次token予測の平均負の対数尤度を指数化した値で、小さいほど良いため、最後のFFNの寄与は大きいと読めます。著者らはこれが出力のtoken分類に関わると解釈していますが、その機能的な説明は実験値そのものとは区別する必要があります。<a class="citation" href="#ref-1">[1]</a>

### 1.3 Calibration dataを変えても、削除層が一致する

ACL 2025掲載版のAppendix Aには、calibration dataを長文書籍のPG19から知識問題のMMLUへ変える実験があります。Tables 9・10の削除層は次の通りで、7B・13Bとも選ばれる集合が完全に一致しています。これはarXiv v3にはない追加結果です。層番号は論文の表記を用い、範囲は両端を含みます。<a class="citation" href="#ref-4">[4]</a>

| モデル | PG19で選んだ層 | MMLUで選んだ層 |
| --- | --- | --- |
| Llama2-7B | 21〜29 | 21〜29 |
| Llama2-13B | 26〜35 | 26〜35 |

この設定では、BIは「その問題分野だけで不要な層」より、データをまたいで共通する変化の小ささを捉えているように見えます。また、手法は連続性を要求しないのに、結果として連続区間が選ばれています。ただし、ここで確かめられたのは二つのcalibration dataと所定の削除数に対する層集合の一致です。全層の順位が同じことや、任意のdomain・モデルでも不変であることまでは示していません。<a class="citation" href="#ref-4">[4]</a>

## 2. The Unreasonable Ineffectiveness of the Deeper Layers：区間を削ってhealingする

### 2.1 一層ずつの変化と、区間全体の変化を分ける

この論文は、連続する \(n\) 層をまとめて削ることを先に決め、その区間の入口と出口が最も似る開始位置を探します。ShortGPTが一層ごとの変化量を順位付けするのに対し、こちらは \(H_\ell\) と \(H_{\ell+n}\) を直接比べます。最終tokenの表現を使った平均角距離と、削除の開始位置は次式です。<a class="citation" href="#ref-2">[2]</a>

<div class="math-scroll">
\[
\begin{aligned}
D_n(\ell)&=\mathbb{E}_{x}\left[
\frac{1}{\pi}\arccos\!\left(
\frac{h_{\ell,T}(x)^\top h_{\ell+n,T}(x)}
{\|h_{\ell,T}(x)\|_2\,\|h_{\ell+n,T}(x)\|_2}
\right)\right],\\
\ell^*(n)&=\operatorname*{arg\,min}_{\ell}D_n(\ell).
\end{aligned}
\]
</div>

\(T\) は各系列の最終位置です。causal attentionでは最終tokenが系列全体を参照できるため、著者らはこの位置を測定対象にしています。選ばれた \(\ell^*,\ldots,\ell^*+n-1\) を削り、元の \(H_{\ell^*}\) を元の第 \(\ell^*+n\) 層へ渡します。したがって、これはBIの小さい層を連続するように集める操作ではありません。削除数 \(n\) ごとに区間全体を評価し直すので、削除数を増やしたときの最適区間が、前の区間を必ず含むとも限りません。<a class="citation" href="#ref-2">[2]</a>

違いは二次元の例で確認できます。二つの層を通った表現が \((1,0)\to(0,1)\to(1,0)\) と変わるなら、各層の入出力のcosine similarityは0で、個別のBIはどちらも1です。しかし区間の入口と出口は一致するため、区間の角距離は0になります。逆に、小さな回転が同じ向きに積み重なると、一層ごとのBIは小さくても区間全体では大きく向きが変わります。つまり、個別変化の小ささと、合成した変換の恒等写像への近さは別の性質です。この例は評価基準の違いを示すもので、実際のLLMでこのような相殺が起きたことを示す実験結果ではありません。

削除後は、必要に応じてQLoRAでhealingします。これは元の重みを4-bitで保持し、残った層のFFNに付けた低ランクadapterを次token予測の損失で学習するものです。削除区間のteacher出力に直接合わせる蒸留ではありません。主実験ではC4を使い、7B以下は約164M token、13B以上は約328M token、5000 step学習します。各実験を40GBのA100一枚で実行できる構成です。ここでMは100万を表します。<a class="citation" href="#ref-2">[2]</a>

### 2.2 「半分削れる」はモデルとタスクを限定した結果

Figure 2の5-shot MMLUでは、しばらく正解率を保った後、ある削除率から急にランダム回答の25%付近へ落ちる傾向が見られます。その境界はモデルによって大きく異なります。以下は本文§4.1が示す概数で、安全に削れる上限を保証する数値ではありません。healingには精度の低下を抑え、崩れる境界をやや先へ延ばす効果があります。<a class="citation" href="#ref-2">[2]</a>

| モデル群 | MMLUが急に崩れる層削減率の目安 |
| --- | --- |
| Llama2 | 約45〜55% |
| Mistral-7B | 約35% |
| Phi-2 | 約25% |
| Qwen | 約20% |

一方、healing後のC4検証損失は、削除率とともに緩やかに増加し、MMLUが急落する境界でも連続的に変化します。次token予測の損失を回復できることと、質問への正解能力を回復できることは一致しません。さらにLlama2-70Bを評価したFigure 6では、数学のGSM8Kと常識推論のHellaSwagは少量の削除から劣化する一方、思考過程を生成させるCoT-MMLUには比較的平坦な領域が残ります。HellaSwagは選択問題なので、生成か選択かだけでもこの差は説明できません。長い出力で誤差が蓄積するという仮説に加え、必要な計算の種類を考える必要があります。著者らは深い層が推論に寄与するという解釈を示していますが、知識と推論を層ごとに完全に分離した証明ではありません。<a class="citation" href="#ref-2">[2]</a>

もう一つ面白いのは、層選択をかなり単純化できることです。最終層だけを残し、その直前から深い層を \(n\) 層削る方法は、healingなしでは性能が大きく落ちます。しかしLlama2-70BのFigure 5では、healing後のMMLU・BoolQ・C4損失が、類似度で区間を選んだ場合に近づきます。削除前のスコアを精密に最適化することと、削除で生じた接続のずれを学習で補うことには、ある程度の代替関係があると読めます。これは学習費用を含めた結果であり、追加学習なしで使う場合に層選択が不要になるわけではありません。<a class="citation" href="#ref-2">[2]</a>

## 3. LLM-Streamline：削除区間の働きを小さなネットワークに移す

### 3.1 区間の入出力を局所的に蒸留する

LLM-Streamlineも連続区間を選びますが、削除後にその区間を軽量ネットワークで置き換えます。区間の選択には全tokenについて平均したcosine similarityを使い、次の開始位置を求めます。前節との違いは、最終tokenだけを見るか全tokenを見るか、そして平均する量が角距離かcosine similarityかです。一つのベクトル対なら後二者は単調な対応を持ちますが、平均を取った後の順位まで必ず一致するわけではありません。<a class="citation" href="#ref-3">[3]</a>

<div class="math-scroll">
\[
\ell^*(n)=\operatorname*{arg\,max}_\ell\mathbb{E}_{x,t}
\left[\cos\!\left(h_{\ell,t}(x),h_{\ell+n,t}(x)\right)\right].
\]
</div>

選んだ区間の入力 \(H_{\ell^*}\) と、元モデルが計算した出力 \(H_{\ell^*+n}\) を保存し、軽量ネットワーク \(g_\phi\) を次の平均二乗誤差で学習します。\(\phi\) はそのネットワークのパラメータ、\(\|\cdot\|_F^2\) は行列の全要素の二乗和です。

<div class="math-scroll">
\[
\phi^*=\operatorname*{arg\,min}_{\phi}
\mathbb{E}_{x}\left[
\frac{1}{Td}
\left\|g_\phi\!\left(H_{\ell^*}(x)\right)-H_{\ell^*+n}(x)\right\|_F^2
\right].
\]
</div>

推論時は、区間を素通りする代わりに \(g_{\phi^*}(H_{\ell^*})\) を後続へ渡します。式の \(g_\phi\) は残差成分だけでなく、置き換え後の出力全体を表しています。残ったLLMを次token予測で調整する前節のhealingと比べると、元の区間が行っていた写像を直接学ぶ点が異なります。入出力を先に保存できるため、軽量ネットワークの学習中にLLM全体の計算グラフを保持する必要もありません。主実験ではSlimPajamaから500サンプルを層選択に、30,000サンプルを置換ネットワークの学習に使い、20 epoch学習します。比較するのはランダム初期化したFFNと、削除区間の最初の層から初期化したTransformer一層です。<a class="citation" href="#ref-3">[3]</a>

3手法の違いを、削除単位と回復方法で対応させると次のようになります。ShortGPTの基本形にも後の版では回復実験がありますが、ここでは中心となる手法を比較しています。<a class="citation" href="#ref-1">[1]</a><a class="citation" href="#ref-2">[2]</a><a class="citation" href="#ref-3">[3]</a>

| 手法 | 削除対象の選び方 | 性能回復 |
| --- | --- | --- |
| ShortGPT | 個々の層のBIを順位付け | 基本形では行わない |
| Deeper Layers | 連続区間の入口・出口の角距離 | 残ったFFNのQLoRA、次token予測 |
| LLM-Streamline | 連続区間の入口・出口のcosine similarity | 代替ネットワーク、区間出力のMSE |

### 3.2 分類性能は回復するが、数学の回復は難しい

Tables 2〜4のLlama2-7Bの結果を示します。分類平均は12タスクの平均正解率です。生成平均はXSumのROUGE-1と、GSM8K・StrategyQAの正解率を平均した論文の集計値で、生成全般を一つの正解率で測った値ではありません。`None` は同論文の区間削除だけの設定であり、ShortGPTや前節のQLoRAを含む実験結果そのものではありません。<a class="citation" href="#ref-3">[3]</a>

<div class="math-scroll" markdown="1">

| 方法 | パラメータ削減率 | 分類平均（%） | 生成平均 | GSM8K（%） |
| --- | --- | --- | --- | --- |
| 元モデル | 0% | 49.2 | 32.0 | 16.5 |
| 削除のみ（None） | 24.0% | 44.0 | 19.5 | 1.97 |
| FFNで置換 | 25.0% | 45.8 | 22.4 | 2.16 |
| Transformer一層で置換 | 24.0% | 45.2 | 24.7 | 1.82 |

</div>

FFNによる置換は、約25%の削減で分類平均の93.1%を維持しています。同表のLLM-Pruner・SliceGPTはともに平均38.4なので、45.8は7.4ポイント高い値です。生成平均ではTransformer一層への置換が元モデルの77.2%を保ちますが、GSM8K単独では16.5%から1.82%へ大きく落ちています。FFNとTransformerの優劣も、分類平均と生成平均では入れ替わります。なお、各設定は代替ネットワークを含む削減率がおおむね揃えられた比較であり、同一の削除区間にmoduleを追加しただけの比較として読むべきではありません。<a class="citation" href="#ref-3">[3]</a>

LoRAとの比較には、学習条件の区別も必要です。Table 6では、局所的なMSE学習に加えて、置換ネットワークを次token予測で1 epoch追加学習しています。この設定のTransformer置換は分類平均46.7%、GPUメモリ27.8GBで、LoRAは44.5%、56.4GBです。ただし、置換側は30,000サンプル、LoRA側は300,000サンプルであり、前節のQLoRAとも別の条件です。また、Appendix E.8で置換後の追加学習データを30,000から549万サンプルへ増やすと、GSM8Kは2.05%から10.6%へ回復しますが、元の16.5%には届きません。15タスク平均の変化は42.5から43.8にとどまる一方、計算時間は約100倍と報告されています。どの能力を回復させたいかによって、追加学習の費用対効果が変わります。<a class="citation" href="#ref-3">[3]</a>

推論速度も改善しています。Appendix Dの128 tokenを生成する測定では、元のLlama2-7Bが19.87 token/s、FFN置換が25.88 token/sで、約1.30倍です。同じ表のSliceGPTは27.20 token/sなので、精度面で良いことと速度で最良であることは一致しません。この節の数値は各論文の評価設定内で比較するもので、ShortGPTの表とモデル名が同じでも、タスク構成や評価方法の異なる平均を直接比べることはできません。<a class="citation" href="#ref-3">[3]</a>

### 3.3 削除する層の安定性と、回答の安定性

ShortGPTのcalibration data比較に対応する結果が、こちらのAppendix Aにもあります。Llama2-7Bから8層を削る際、cosine similarityではSlimPajama・WikiText・C4のいずれを使っても層22〜29が選ばれます。一方、一層ずつ削ったときのPPLを測って選ぶ手順では、データによって削除層が変わります。Table 8の値は次の通りです。<a class="citation" href="#ref-3">[3]</a>

<div class="math-scroll" markdown="1">

| Calibration data | PPLによる削除層 | Cosine similarityによる削除層 |
| --- | --- | --- |
| SlimPajama | 9, 10, 11, 12, 21, 23, 25, 27 | 22〜29 |
| WikiText | 9, 10, 11, 12, 21, 23, 24, 27 | 22〜29 |
| C4 | 8, 9, 11, 12, 22, 23, 24, 25 | 22〜29 |

</div>

さらにTable 9では、SlimPajamaで選んだモデルの同データ上のPPLは、PPL基準が12.1、cosine基準が19.7で、前者が良い値です。しかしMMLUは25.9%対44.6%、15タスク平均は35.0対39.1で、下流性能は後者が上回ります。削除を決めるデータ上の言語モデル損失を小さくすることが、能力の維持と同義ではないわけです。ただし、この比較はスコアだけでなく、逐次的な層削除と連続区間の選択という探索方法も異なります。cosine similarityの普遍的な優位性や、任意のdomainへの頑健性まで結論することはできません。<a class="citation" href="#ref-3">[3]</a>

削除層が安定していても、回答まで安定しているとは限りません。Llama2-13Bでは削除のみの分類平均54.5%がFFN置換の53.2%を上回る一方、元モデルとの正誤の一致を測るstabilityは81.1%対84.9%、生成平均は22.0対28.4で、FFN置換が良い値です。正解数だけを集計すると、元々正解していた問題を間違える変化と、元々間違えていた問題に正解する変化が相殺されます。そのため著者らは、圧縮前後の正誤の一致を元モデルの確信度で重み付けするstabilityを併記します。ただし、この指標にも元モデルの誤りを直すと一致度が下がるという性質があります。能力の高さと元モデルへの忠実さは別々に評価すべきであり、stabilityの厳密な定義はAppendix Aに置きます。<a class="citation" href="#ref-3">[3]</a>

## Appendix A. Stabilityは「同じ回答」ではなく「正誤の一致」を測る

LLM-Streamlineのstabilityを、添字を揃えて書くと次式です。\(a_i,\tilde a_i\in\{0,1\}\) はそれぞれ元モデルと圧縮モデルが問題 \(i\) に正解したかを表します。\(s_i\) は元モデルで測った選択肢間のPPLの標本標準偏差、\(N\) は問題数です。論文はこのばらつきを確信度の代理として用います。<a class="citation" href="#ref-3">[3]</a>

<div class="math-scroll">
\[
\mathrm{Stability}
=\frac{\sum_{i=1}^{N}\exp(s_i)\,\mathbf{1}[a_i=\tilde a_i]}
{\sum_{i=1}^{N}\exp(s_i)}.
\]
</div>

両方正解と両方不正解を一致として数えるので、異なる誤答を選んでいても両方不正解なら一致です。これは回答ラベルそのものの一致率ではありません。また、選択肢のPPLのばらつきは正解確率として校正された確信度ではなく、指数重みは大きなばらつきのサンプルを強く重視します。論文本文には指数関数で重みの差を緩和するという説明がありますが、その説明は通常の指数関数の性質と整合しません。本稿では式に従って解釈しています。<a class="citation" href="#ref-3">[3]</a>

## Appendix B. ACL掲載版のShortGPT-genは、生成時には全層を使う

ShortGPTのACL 2025掲載版には、arXiv v3にはないShortGPT-genも追加されています。入力tokenはBIの小さい層を飛ばし、生成するtokenは全層を通す方法です。Table 3のLlama2-13Bでは、GSM8Kは元モデル28.89%、通常のShortGPTで2.35%、ShortGPT-genで26.91%です。この結果は、入力処理と生成処理で層の必要性が異なることを考える材料になります。ただし、生成時に使う層は保持する必要があるため、全段階で層を物理的に削る圧縮と同じメモリ削減にはなりません。本文のarXiv v3の実験値と、この掲載版の評価値は区別しています。<a class="citation" href="#ref-4">[4]</a>

## 参考文献

<ol class="references">
  <li id="ref-1">Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen. <cite><a href="https://arxiv.org/abs/2403.03853v3">ShortGPT: Layers in Large Language Models are More Redundant Than You Expect</a></cite>. arXiv:2403.03853v3, 2024. 初稿2024年3月6日、参照版2024年10月11日。§§2–4、Tables 1–2、Appendix D.</li>
  <li id="ref-2">Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts. <cite><a href="https://arxiv.org/html/2403.17887v2">The Unreasonable Ineffectiveness of the Deeper Layers</a></cite>. arXiv:2403.17887v2, 2025. 初稿2024年3月26日、参照版2025年3月3日。§§3–5、Figures 1–6、Appendices A–B.</li>
  <li id="ref-3">Xiaodong Chen, Yuxuan Hu, Jing Zhang, Yanling Wang, Cuiping Li, Hong Chen. <cite><a href="https://arxiv.org/pdf/2403.19135v5">Streamlining Redundant Layers to Compress Large Language Models</a></cite>. ICLR 2025, arXiv:2403.19135v5. 初稿2024年3月28日、参照版2025年1月25日。§§2–4、Tables 2–6、Appendices A、C–D、E.8.</li>
  <li id="ref-4">Xin Men, Mingyu Xu, Qingyu Zhang, Qianhao Yuan, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen. <cite><a href="https://aclanthology.org/2025.findings-acl.1035/">ShortGPT: Layers in Large Language Models are More Redundant Than You Expect</a></cite>. Findings of ACL 2025, pp. 20192–20204. §3.3、Table 3、Appendix A、Tables 9–10。arXiv v3と異なる追加実験を参照。</li>
</ol>
