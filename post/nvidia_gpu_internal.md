---
title: [WIP] NVIDIA GPUのお勉強メモ
date: 2026-10-11
author: Hirokazu Ishida
description: CUDAのkernel・grid・block・warp・threadの関係から、SM上のレジスタとshared memoryまで、NVIDIA GPUの基本的な仕組みを整理する。
---

CUDAを勉強し始めたのですが、SMやwarp、register fileの定義を読んでも、互いの関係がよくわからず、なかなか覚えられません... CUDA Programming Guide等の資料にはそれぞれの定義が載っていますが、「なぜその単位に分かれているのか」が見えてきませんでいた。このメモでは、それぞれの関係性や制約を意識しながら、GPUの構成要素や概念をまとめていきます。

## 1. GPUの処理単位

### 1.1 階層

CUDAでは、GPU上で並列実行する処理の入口となる関数をkernelと呼びます。たとえば1次元の起動で `kernel<<<num_blocks, threads_per_block>>>(...)` と書くと、`num_blocks` 個のblockが作られ、それぞれに `threads_per_block` 個のthreadが含まれます。各threadは同じkernelを実行しますが、threadのIDやデータに応じて、異なる処理を担当できます。この1回の起動に含まれるblock全体がgridです。kernelが「何を実行するか」を表すのに対して、gridは「その処理を実行するthread全体のまとまり」を表します。大きい単位から並べると、次のようになります。[Programming Model](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#programming-model)

| 単位 | 役割 |
| --- | --- |
| Grid | 1回のkernel起動に含まれるblock全体 |
| Thread block | 同じSMに配置され、shared memoryやblock内同期で協調できるthreadのまとまり |
| Warp | block内のthreadを32個ずつに分けた、命令発行・実行のグループ |
| Thread | 個々の処理を実行する単位。レジスタの値はthreadごとに持つ |

warpはblock内の連続するthread IDから構成されます。たとえば64 threadのblockなら2 warpです。blockのthread数が32の倍数でなければ、最後のwarpには使われない枠ができます。

### 1.2 Block：SM内で協調する単位

SM（Streaming Multiprocessor）は、GPU内に複数ある演算ユニットです。それぞれがregister file、shared memory、warp schedulerなどの資源を持っています。ひとつのblockは、全体が同じSMに配置されます。そのblock内のthreadは、shared memoryでデータを共有し、`__syncthreads()` で同期できます。つまり、blockはSM内の資源を使ってthread同士が協調する単位といえます。

ただし、blockとSMが1対1に対応するわけではありません。資源に余裕があれば、ひとつのSMに複数のblockが同時に常駐します。また、grid内のblock数はSM数より多くても構いません。blockを受け入れる余裕ができたSMで、残りのblockが順次実行されます。[Hardware Multithreading](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#hardware-multithreading)

### 1.3 Warp：命令をまとめて発行する単位

blockが協調の単位だとすると、warpを設ける理由は何でしょうか。threadごとにレジスタの値は独立しているのに、なぜ命令の実行をまとめるのでしょうか。warpでまとめる利点は、複数のthreadで命令の取得・解釈・発行の仕組みを共有できることだと考えています。もしwarpがなく、32threadがそれぞれ独立に"並列に"動いていたとするならば、それぞれが命令をfetch/decode等する回路を持つ必要がありますし、32個の命令を同時に供給するためのbusも必要になります。こうしたものを一つにまとめられ回路規模を小さくできるのがwarpの利点だと考えています。一点注意したいのが、warpを使う場合も分岐が生じれば並行処理をすることもあるという点です。ただし"並列"ではないため、同時に発行する命令は一つです。

ここで、逆に、命令を共有するならblock全体をまとめて(巨大なwarpとして)実行すればよいのでは、という疑問が生じるかもしれません。ここで気にすべきは、まとめる範囲が広がるほど、異なる分岐や待ち状態を一緒に扱うことになる点です。たとえば、64 threadのblockに `if (...) A; else B;` という分岐がある場合を考えます。

| 分岐の仕方 | Warpごとの実行 |
| --- | --- |
| warp 0の全threadがA、warp 1の全threadがBを選ぶ | 各warpはそれぞれの経路だけを実行できる |
| ひとつのwarp内でAとBを選ぶthreadが混在する | AとBの経路を、それぞれ該当するthreadだけ有効にして実行する |

後者がwarp内の分岐です。別々のwarpなら、それぞれ異なる分岐先を独立に実行できます。また、あるwarpがメモリ待ちでも、schedulerは別の実行可能なwarpの命令を選べます。block全体をひとつの命令発行単位にするよりも、細かく実行を進められるわけです。[SIMT Architecture](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#simt-architecture)

まとめるとwarpという単位および32という値は、命令をまとめて発行する効率と、異なる処理を進める柔軟さのトレードオフを勘案してNVIDIAが採用したものと考えるのがよさそうです。

## 2. レジスタについて

処理単位の関係を押さえたところで、次は各threadの値を保持するレジスタを見てみます。以下では、通常の静的なレジスタ割り当てを対象にします。

### 2.1 コンパイル時と実行時の「レジスタ」

「レジスタ」には、PTXの仮想レジスタ、SASSのレジスタ、ハードウェア上のregister fileという、異なる段階の話が出てきます。コンパイル過程で使われるPTXは、仮想的な命令セットです。仮想レジスタがそのままハードウェア上に確保されるのではなく、対象GPU向けの機械語へ変換する際に、レジスタの割り当てなどが行われます。その機械語を逆アセンブルしたものがSASSです。ここにはGPR（General Purpose Register、汎用レジスタ）が `R0`、`R1` などの名前で現れます。同じ名前でも、値はthreadごとに独立しています。実行時には、blockを配置するSMのregister fileから必要な容量を確保します。コンパイラによるレジスタの割り当てと、blockを受け入れるためのSM上の資源確保は、別の段階です。通常の静的な割り当てでは、1 threadあたりのレジスタ数はコンパイルされたカーネルごとに決まり、各threadに同じ数が割り当てられます。軽い分岐を通るthreadだけ、確保量が減るわけではありません。

### 2.2 レジスタの確保量には、割り当て単位がある

1 threadが使うレジスタ数を \(r\)、block内のthread数を \(T\) とすると、block全体で必要な量は、まず \(rT\) 個と考えられます。ここでは32 bitのレジスタを1個として数えます。ただし、実際の確保量には、ハードウェアの割り当て単位による切り上げがあります。たとえばCompute Capability 7.0では、warpごとに256個単位で切り上げます。1 threadあたり9個使う場合、32 threadのwarpに必要なのは \(9 \times 32 = 288\) 個ですが、確保されるのは512個です。[Calculating Occupancy](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#calculating-occupancy)

<div class="equation math-scroll">
\[
R_{\mathrm{warp}} = 256\left\lceil\frac{32r}{256}\right\rceil
\]
</div>

差の224個分は、確保した領域の内側に使い切れない部分ができる内部断片化と捉えられます。そのため、使用量を少し減らしても確保量は変わらず、切り上げの境界をまたいで初めて減る場合があります。なお、256個という単位はCompute Capability 7.0の例で、GPUによって割り当ての規則は異なります。

### 2.3 レジスタ使用量と、常駐できるwarp数

各SMのregister fileは、常駐するblockで分け合います。ひとつのblockの確保量が増えるほど、同じSMに置けるblock数が減る可能性があります。ただし、次の2つは別の状況です。

- 1 block分は収まるが、複数block分は収まらない。 起動は可能でも、SMに同時に常駐できるblock数が制限されます。
- 1 block分すら確保できない。 指定した構成ではblockを実行できず、そもそもカーネルの起動が失敗します。

順番に実行すれば済むのは前者で、後者では最初の1 blockを置けません。実際の常駐数は、shared memoryの使用量やthread数・block数の上限にも制約されます。常駐するwarp数の、SMが収容できる最大warp数に対する割合をoccupancyと呼びます。レジスタ使用量は、この常駐可能なwarp数を制限します。常駐とは実行状態をSM上に保持していることで、全warpが同じ瞬間に命令を実行するという意味ではありません。

ただ、occupancyを上げれば必ず速くなるわけではありません。レジスタ数を無理に制限すると、値をlocal memoryなどへ退避するspillが増えることもあります。これはコンパイラが生成する処理で、起動時の資源不足を自動的に解消する仕組みではありません。

### 2.4 レジスタの物理的な配置に関する未確認事項
ちょっと気になったのが、register file内の物理的な配置です。連続した領域を必要とするのか、バラバラの領域をつなぎ合わせて割り当てることができるのか、どちらなのか興味がありますが、まだ確認できていません。後者の場合は特段問題はありませんが、前者の場合は、断片化している場合にトータルとしては十分な空き容量があっても、連続した領域が確保できずに割り当てられないことがあるのではないでしょうか。

### Appendix A. Shared memoryとL1 cacheの違い

shared memoryはプログラマが領域と内容を明示的に管理します。一方、L1 cacheはメモリアクセスに応じてハードウェアが内容を管理します。

A100では、L1・texture・shared memoryがSMあたり192 KBの容量を共有し、shared memory側の容量を対応する設定値から選びます。shared memoryを多く確保する設定ほどL1側の容量は減りますが、blockが要求したbyte数をそのまま引く関係ではありません。[Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.1/ampere-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)

アクセス条件によってレイテンシも変わるため、「shared memoryはL1より何cycle速い」という固定値ではなく、まずは管理方法と役割の違いで区別します。
