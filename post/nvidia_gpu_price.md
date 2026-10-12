---
title: [WIP] NVIDIAのGPUはなぜ高いのか——価格・競争・供給制約の関係
date: 2026-10-13
author: GPT-6-Astra xhigh（構成・執筆）／石田寛和（質問攻め・編集）
description: GPUの価格を、代替製品への移行、製造能力の制約、クラウド利用料と小売価格の違いから考える。
---

NVIDIAのAI向けGPUは、なぜ高いのでしょうか。高度な半導体だから製造にお金がかかる、という説明には納得できます。しかし、それだけでは、競合が安い製品を出すことで価格が下がらないのか、という疑問が残ります。高く売れるのなら、NVIDIAや製造を担う企業が増産すればよいのでは、とも考えられます。この記事では主にデータセンター向けGPUを対象に、他社GPUやAI専用チップへの移行と供給能力の拡大を難しくする要因から、この疑問を整理します。さらにH100・A100のクラウド利用料とRTX 5090の小売価格を見て、「GPUの価格」がどの取引の価格を指すのかを区別します。

価格と原価の関係を考える手がかりとして、NVIDIAの2027会計年度第2四半期、2026年7月26日に終了した四半期の全社売上総利益率は75.0%でした。売上から売上原価を引いた額が、売上の約4分の3に相当するということです。ただし、これはGPU一基の利益率ではありません。研究開発費などは別途かかり、売上原価も追加の一基を作る費用とは異なります。この数字から個別製品の製造原価を逆算することはできませんが、少なくとも価格の説明を製造費用だけで終えるわけにはいきません。<a class="citation" href="#ref-1">[1]</a>

## 1. 他社がもっと安く提供できないのか

### 1.1 CUDAからの移行には費用がかかる

安い計算資源があっても、乗り換えに費用がかかれば、その価格差は相殺されます。NVIDIAには、GPUの開発基盤であるCUDAと、周辺のライブラリ・開発ツールの蓄積があります。<a class="citation" href="#ref-2">[2]</a> 別のGPUやAI専用チップへ移るには、これらに依存するコードの移植、検証、性能調整が必要です。例えばAMDのHIPIFYはCUDAのコードをHIPというGPUプログラミング環境向けに変換しますが、未対応のライブラリの置き換えや実機での検証までは省けません。<a class="citation" href="#ref-3">[3]</a> このスイッチングコストが、NVIDIAの価格を支える一因です。比較すべきなのはチップの単価だけでなく、同じ処理を終える費用と、そこへ移行する費用です。

### 1.2 AMD GPU・Google TPU・AWS Trainiumという選択肢

代替はAMDのGPUに限りません。GoogleのTPU（Tensor Processing Unit）やAWSのTrainiumは、機械学習の計算に向けて設計された独自チップです。<a class="citation" href="#ref-4">[4]</a><a class="citation" href="#ref-5">[5]</a> すでに実際のモデル開発・提供に使われており、Anthropicは2026年4月、Claudeの学習と実行にTPU、Trainium、NVIDIA GPUを使っていると説明しています。<a class="citation" href="#ref-6">[6]</a> NVIDIA以外でも計算できるかという段階から、どの処理をどの基盤で動かすと有利かを選ぶ段階に進んでいる企業があるわけです。

これを支えるのは、チップとソフトウェアの両方の開発です。AMDにはROCmやHIP、GoogleのTPUにはJAXなどから使うXLAコンパイラ、AWSにはコンパイラや実行環境を含むNeuronがあります。<a class="citation" href="#ref-7">[7]</a><a class="citation" href="#ref-4">[4]</a><a class="citation" href="#ref-5">[5]</a> 利用者も開発に参加しています。MetaやOpenAIはAMD向けの最適化を進め、AnthropicはTrainium向けの低レベルkernelを書き、Neuronにも貢献しています。<a class="citation" href="#ref-8">[8]</a><a class="citation" href="#ref-9">[9]</a><a class="citation" href="#ref-10">[10]</a> フレームワークへの対応だけでなく、実際の処理を効率よく動かす部分まで整えることで、代替が使いやすくなります。

複数のチップでコードを共用する取り組みもあります。例えばMetaとGoogleは2026年7月、kernelを記述する言語HelionからTPU向けのコードを生成する仕組みを紹介しました。<a class="citation" href="#ref-11">[11]</a> PyTorchなどのモデル記述を共通化する取り組みと、kernelの実装を共通化する取り組みは、いずれも移行費用を下げる方向に働きます。ただし、対応するAPIがあることと、既存のコードが変更なしに同じ性能で動くことは別です。ソフトウェアの役割の違いや低レイヤーの開発例は、<a href="#appendix-alternative-software">Appendix F</a>にまとめます。

### 1.3 AI coding agentは移行費用を下げられるか

こうした開発をさらに後押しする可能性があるのが、コードの生成・修正・検証を繰り返すAI coding agentです。僕自身、CUDAやアセンブリを自動生成するコンパイラなどをAI codingで書いたことがありますが、かなり手軽に実装できました。この手軽さが別のGPUやAI専用チップにも及べば、共通基盤で足りない部分を自分たちで補う費用も下がります。

実例もあります。MetaのKernelEvolveは複数のハードウェアを対象にkernelを生成・最適化し、AMDのGEAK v4もHIPやTritonのコードを生成して実機で検証します。<a class="citation" href="#ref-12">[12]</a><a class="citation" href="#ref-13">[13]</a> AWSも2026年7月、PyTorchやTritonのkernelをTrainium向けのNKI（Neuron Kernel Interface）へ書き換え、実機でコンパイル・検証・性能測定を繰り返すagentを公開しています。<a class="citation" href="#ref-14">[14]</a> 対象は個別のkernelで、アプリケーション全体の移行や性能を保証するものではありません。

個人の開発にも例があります。Anthropic所属のSasha Krassovsky氏が公開するSporkでは、Apple GPU向けの行列積やattention kernelを追加・修正した2026年5月のコミットに、Claudeの共著者記録が残っています。<a class="citation" href="#ref-15">[15]</a><a class="citation" href="#ref-16">[16]</a><a class="citation" href="#ref-17">[17]</a><a class="citation" href="#ref-18">[18]</a> 共著者表記だけからAIの担当範囲や工数削減率はわかりませんが、NVIDIA以外のkernel開発にもAIが関与している具体例です。共通基盤の整備に、残る専用コードを書く費用の低下が加われば、NVIDIA以外を選べる条件は広がると考えています。

ただし、代替チップへ処理を移せることと、半導体を増産できることは別です。既設のTPUやTrainiumなどに空きがあれば不足を緩和できますが、新たにチップを作る場合、同じ不足工程を共有していれば製造能力の制約は残ります。選択肢が増えることと、供給全体が需要に追いつくことを分けて考える必要があります。

## 2. なぜ増産しても、すぐに不足を解消できないのか

強い需要を見込んだ増産投資は、実際に進んでいます。TSMCは2026年7月、AIを含む顧客の需要の強さを理由に、同年の設備投資予算を600〜640億米ドルへ引き上げました。<a class="citation" href="#ref-19">[19]</a> 半導体製造装置を供給するASMLも、同月の決算説明で、顧客の半導体メーカーが積極的に生産能力を増やしていると報告しています。<a class="citation" href="#ref-20">[20]</a> そこで考えるべきなのは、投資を決めてから供給が増えるまでの時間と、どこまで設備を増やすかという判断です。

### 2.1 不足する工程の増設には時間がかかる

NVIDIAは演算用チップの製造をTSMCなどへ委託し、HBM（High Bandwidth Memory）という広帯域メモリの調達や、製品の組み立ても外部の企業に頼っています。<a class="citation" href="#ref-2">[2]</a> AMDも7nm以下のプロセスを使うGPUのウェハー製造をTSMCに依存しており、販売元が変わっても、その背後に独立した製造能力があるとは限りません。<a class="citation" href="#ref-21">[21]</a> 増産するには、チップ、メモリ、組み立てのうち、不足する部分の能力を増やします。HBMの構造と、他のメモリとの生産配分については<a href="#appendix-hbm">Appendix A</a>にまとめます。

既存設備の稼働率や生産配分の変更で足りなければ、設備を増設します。既知の製造方法を使う増設でも、装置を設置し、必要な品質と生産量を出せる状態まで立ち上げる作業は残ります。Micronは設備拡張を、専門的な人材や装置を必要とする複数年の事業と説明しています。同社がシンガポールのHBM向け先端パッケージング施設に着工したのは2025年1月で、生産能力を本格的に拡張する計画は2027年初めからです。<a class="citation" href="#ref-22">[22]</a> 工事の並行化や既存施設の転用によって期間を短縮する動きもありますが、建物の完成と製品の量産開始は別の段階です。各社の事例は<a href="#appendix-expansion">Appendix B</a>にまとめます。

### 2.2 技術改善で生産性を上げる

製造方法を変えて供給を増やす取り組みもあります。微細化によってウェハー一枚から得られるメモリ容量を増やす、工程を短縮する、良品の割合である歩留まりを上げる、といった改善です。具体例は<a href="#appendix-productivity">Appendix C</a>で扱います。ただし、技術改善だけで需要増を吸収できるとは限りません。Micronは2026年6月、製造世代の移行によるビット単位の生産量の伸びが鈍化し、供給拡大には工場の新設も必要になっていると説明しています。<a class="citation" href="#ref-23">[23]</a> また、メモリの製造が速くなっても、組み立て工程が不足すればGPUの出荷はそこで制限されます。生産性の改善と設備増設を組み合わせ、製品になるまでの各工程の能力をそろえる必要があります。

### 2.3 設備が完成する頃の採算を考える

さらに、増設の規模を決めるには、供給が増えた後の価格で採算を考える必要があります。建設と立ち上げに時間がかかるほど、投資判断は遠い将来の需要と価格の予測に依存します。しかも、その間に競合も設備を増やします。需要の成長を見込めても、現在の高値で追加生産分をすべて売れるとは限りません。ある規模の増産が有望でも、さらに設備を増やすほど利益も増えるとは限らないわけです。

説明用の仮定として、同じ性能の製品について、現在の年間需要量と年間生産能力をともに100とします。設備が完成する頃に、同じ価格での需要が130へ伸びても、各社の増設を合計した能力が160になれば、その価格では全量を売れません。価格を下げて販売量を増やすか、設備の一部を稼働させないことになります。需要が伸び続けても、供給の伸びがそれを上回れば、価格や稼働率は下がりえます。AI需要の急減が起きなくても、増設後の採算が悪化する場合はある、ということです。

### 2.4 長期購入契約で不確実性を減らす

この不確実性を減らす手段の一つが、将来の購入量や価格条件を先に取り決める長期購入契約です。Micronは2026年9月30日の決算説明で、複数年の長期顧客契約を26件締結済みと報告しています。購入数量に拘束力を持たせ、価格の上下限を設けた契約もあります。同社は、こうした契約による需要の見通しの改善を、2027会計年度の設備投資計画の増額を支える要因に挙げています。<a class="citation" href="#ref-24">[24]</a> ASMLも、顧客企業が販売先と結ぶ長期契約が、能力増強への確信を支えていると説明しています。<a class="citation" href="#ref-20">[20]</a> 買い手にも購入を約束してもらうことで、増設の判断をしやすくするわけです。ただし、契約の不履行などのリスクは残り、設備の建設時間もなくなりません。<a class="citation" href="#ref-22">[22]</a>

## 3. 実際に上がっているのは、誰が売る何の価格なのか

### 3.1 小売価格とクラウド利用料を分ける

NVIDIAがGPUを売る価格、販売店が完成品を売る価格、クラウド事業者が計算資源を貸す料金は、別々に決まります。NVIDIAにはクラウド事業者への直接販売と、サーバーメーカーなどを介した間接販売があり、調達経路も一つではありません。<a class="citation" href="#ref-2">[2]</a> クラウド利用料が上がったからといって、NVIDIAがGPUを値上げしたとは限りません。個別顧客へのH100・H200の納入価格の推移は、今回の資料からは確認できません。

利用者が払う価格にも、異なる動きがあります。RTX 5090搭載の同一製品を追うと、国内掲載最安価格は2025年8月1日の38万円から2026年10月13日の112万9,800円へ、約3倍になりました。<a class="citation" href="#ref-25">[25]</a> 一方、NebiusのH100の通常利用料は、1基・1時間あたり2024年8月の3.50米ドルから2026年10月の4.50米ドルへ上がっていますが、2023年の提供開始時の4.85米ドルは下回っています。<a class="citation" href="#ref-26">[26]</a><a class="citation" href="#ref-27">[27]</a><a class="citation" href="#ref-28">[28]</a> 「当初より安い」と「最近は値上がりした」は両立します。RTX 5090の数字も一製品の掲載価格で、実際の売買価格や市場全体の平均ではありません。A100やH200・B200を含む履歴と比較条件は、<a href="#appendix-prices">Appendix D</a>にまとめます。

### 3.2 品薄の影響は、新品と既設設備で異なる

RTX 5090では、部品供給の制約と小売店の品薄が確認されています。NVIDIAは2026年1月、GeForce RTXの需要が強く、メモリ供給に制約があると回答しています。<a class="citation" href="#ref-29">[29]</a> 同月の秋葉原の店舗取材でも、RTX 5090などは入荷してもすぐ売れる状況でした。<a class="citation" href="#ref-30">[30]</a> 新品の補充が需要に追いつかなければ、高値でも売れる余地が生まれます。ただし、先ほどの約3倍という上昇のうち、部品の原価と品薄による上乗せがそれぞれどれだけを占めるかはわかりません。

クラウドでは、増設が遅れても、稼働中のGPUは計算資源の供給を続けられます。新品の入荷量と、既設設備を含む計算サービスの供給量は同じではなく、価格への影響も異なる一因だと考えています。ただし、既設GPUに余力がなくなれば利用料も急騰し得ます。AWSのCapacity Blocksも、需給に応じて新規予約の料金を更新する一方、購入済みの予約には購入時の料金を適用します。<a class="citation" href="#ref-31">[31]</a> 供給不足の影響は、残る空き容量と契約条件によって変わるわけです。

### 3.3 大口契約は、供給の取り合いと増設の両方に効く

OpenAIやAnthropicは巨大な需要源です。例えばAnthropicは2025年11月、NVIDIA GPUを使うAzureの計算資源を300億ドル分購入する約束を発表しました。<a class="citation" href="#ref-32">[32]</a> これは将来の利用を含む契約で、同額のGPUをその場で買ったという意味ではありません。ほかにも大口の利用者がおり、両社が世界の供給の何割を確保しているかは今回の資料からは算定できません。「両社の買い占めが不足の主因」とまではいえず、調達規模を数える際の注意点は<a href="#appendix-compute-buyers">Appendix E</a>で補います。

供給に余裕が少なければ、大口顧客の追加予約がほかの利用者の枠を減らし、価格や納期を押し上げることは考えられます。一方、将来の増設分を購入する約束は、2.4節で見たように、設備を作る側に買い手の見通しを与えます。短期的な供給の取り合いを強める面と、長期的な増設を支える面の両方があります。


<a id="appendix-hbm"></a>

## Appendix A. HBMと通常DRAMの生産配分

HBMは、DRAMの半導体チップであるダイを積み重ね、シリコンを貫通する電極であるTSV（Through-Silicon Via）で接続したメモリです。SK hynixの技術解説では、TSVが占める面積や、積層時の歩留まりも設計上の課題として挙げられています。<a class="citation" href="#ref-33">[33]</a> この積層・接合と、完成したHBMをGPUの演算用チップと一つのパッケージに組み合わせるCoWoSは別の工程です。<a class="citation" href="#ref-34">[34]</a> HBMの供給を考える際には、ダイの製造と、その後の積層・接合の能力を区別する必要があります。

通常のDRAMと同じ容量を作る場合も、製造に必要な資源は同じではありません。Micronは2026会計年度の年次報告書で、同じ製造世代・同じビット数で比べると、HBMには通常のDRAMより多くのウェハーとクリーンルーム面積が必要だと説明しています。また、HBM向けの能力を通常DRAMへ振り替えると、通常DRAMの供給が増える可能性にも言及しています。<a class="citation" href="#ref-22">[22]</a> これは、両者の生産枠を完全に独立して増やせるわけではなく、配分を変えられる部分があることを示します。

既存設備がフル稼働し、生産効率も一定なら、HBM向けの生産枠を増やすぶん、通常DRAM向けの枠は減ります。ただし、完成した通常DRAM用のダイをHBMに変えるという意味ではなく、これから製造する製品の配分を変えるという話です。HBMへの振り替えは増産の手段になりますが、積層・接合の能力も足りている必要があります。設備を増設する判断と、限られた設備をどの製品に割り当てるかの判断は、分けて考える必要があります。

<a id="appendix-expansion"></a>

## Appendix B. 増設の期間はどこまで短縮できるのか

工期を短縮する工夫は、すでに行われています。SK hynixの建設部門は、設計ができた部分から着工し、設計と工事を並行させる「ファストトラック」を採用しています。2025年の技術者インタビューでは、過去のM11工場で、10か月で建物を完成させクリーンルームを稼働させた事例が紹介されています。これは量産開始までの期間ではありませんが、すべての作業を順番に終えるまで待つわけではないことがわかります。同じ記事では、M15X工場でも建築工法を改良し、クリーンルームを早く使えるようにする取り組みが説明されています。<a class="citation" href="#ref-35">[35]</a>

一方、人数を増やすには、必要な技能を持つ人を確保する必要があります。TSMCは2023年7月、アリゾナ第1工場で、半導体工場への装置設置に専門技能を持つ作業員が不足していると説明し、台湾から経験のある技術者を送り、現地の作業員を訓練する方針を示しました。<a class="citation" href="#ref-36">[36]</a> これは海外での新工場建設の事例であり、既存拠点の増設にも同じ問題が生じるとは限りません。それでも、工事費と総人数だけで工期が決まるわけではないことを示します。

建物を再利用して時間を縮める例もあります。Micronは2026年3月、台湾のPSMCからTongluoの工場を取得し、既存クリーンルームを改修して先端DRAMの供給を増やす計画を発表しました。<a class="citation" href="#ref-37">[37]</a> 同年9月時点では、まとまった製品出荷の開始を2027年半ばと見込んでいます。また、新しいDRAM・NAND工場については、最初の生産から数四半期かけて生産量が増えていくと説明しています。<a class="citation" href="#ref-24">[24]</a> 既存施設を使う場合でも、取得した翌日から必要な製品を大量に出荷できるわけではありません。ただし、他社工場の改修と自社の既存工程の増設では条件が異なり、この計画から増設一般の最短期間を求めることはできません。

これらの事例からは、工事の並行化、熟練者の投入、施設の再利用で短縮できる部分と、装置の導入や生産量の立ち上げに時間を要する部分を分けて考える必要があります。製造方法が既知であることは増設の助けになりますが、それを新たな設備で実現する作業までなくなるわけではありません。

<a id="appendix-productivity"></a>

## Appendix C. 製造方法を変えて供給を増やす

Micronの1γ（1-gamma）は、ウェハー一枚から得られるメモリ容量を増やす例です。同社は2025年2月、1γを使った16Gb DDR5のサンプル出荷を発表し、前世代の1βと比べてウェハーあたりのビット数が30%超増えると説明しました。極端紫外線を使うEUV露光と、設計・製造工程の改良によるものです。<a class="citation" href="#ref-38">[38]</a> ウェハーの投入枚数を増やす以外にも、容量の供給を増やす手段があるわけです。ただし、この比較をHBMにそのまま適用したり、既存設備を変えずに工場全体の生産量が30%増えると読んだりすることはできません。

TSMCのN6には、工程を簡略化して生産性を上げる例があります。同社は、EUV露光を適用する層を増やすことで工程を簡略化し、製造にかかる時間を短縮していると説明しています。また、N7の設計を同じダイ面積でN6へ移す場合にも、必要なマスク数の削減と工程の簡略化によって歩留まりが改善するとしています。<a class="citation" href="#ref-39">[39]</a> 小さなチップを多数取るだけでなく、同じ大きさのチップを、少ない工程で良品にする方向の改善です。N6は2020年に量産を始めた世代であり、これは製造技術の効果を示す例として挙げています。

HBMの積層にも改善があります。SK hynixのMR-MUF（Mass Reflow-Molded Underfill）は、チップを積み重ねた後に隙間へ液状の保護材料を入れ、硬化させる方法です。同社は、チップを積むたびにフィルム状の材料を挟む方式と比べて工程の効率がよく、放熱性にも利点があると説明しています。2023年に発表した12層のHBM3にも、この方法の改良版を使っています。<a class="citation" href="#ref-40">[40]</a> ウェハー上に回路を作る工程だけでなく、できたチップを積層する方法も、生産性を改善する対象です。

ただし、これらの改善率を足して、GPU全体の増産率とすることはできません。メモリ容量、製造時間、歩留まりはそれぞれ違う指標であり、改善した工程以外が供給を制約している場合もあります。技術で供給を増やす努力は実際に進んでいますが、不足の解消にどれだけ効くかは、改善の大きさと、その時点で不足している工程によって変わります。

<a id="appendix-prices"></a>

## Appendix D. GPU価格の履歴と比較条件

### D.1 H100・A100の事業者別料金

NebiusのH100の公表料金は、次のように動いています。単位は1基・1時間あたりの米ドルです。提供開始時の料金、値下げの発表、現在の料金表という三つの観測点であり、途中の全改定を追ったものではありません。<a class="citation" href="#ref-28">[28]</a><a class="citation" href="#ref-26">[26]</a><a class="citation" href="#ref-27">[27]</a>

| 公表時期・確認時点 | H100の従量課金 |
| --- | ---: |
| 2023年11月・提供開始 | 4.85米ドルから |
| 2024年8月・値下げ報告 | 3.50米ドル |
| 2026年10月13日・料金表を確認 | 4.50米ドル |

AWSにも値下げの例があります。2025年6月1日、A100搭載のP4d・P4deのオンデマンド料金は33%、H100搭載のP5は44%引き下げられました。基準は同年5月31日の料金で、対象インスタンスを提供する全リージョンへの適用です。<a class="citation" href="#ref-41">[41]</a> 一方、LambdaのA100 40GB・1基構成は、2023年公開・同年8月更新の記事では1時間1.10米ドル、2026年10月13日の通常料金はSXM版・PCIe版ともに税別1.99米ドルです。<a class="citation" href="#ref-42">[42]</a><a class="citation" href="#ref-43">[43]</a> ただし、CPUなどの周辺構成まで同一とは確認できません。これらを一本の時系列につなげたり、A100市場全体の高騰と読んだりせず、事業者、構成、契約条件をそろえて比較する必要があります。

### D.2 H200・B200の利用料も上昇した例がある

Silicon DataのNeo-Cloud指数は、GPU向けクラウド事業者群の料金を集計したものです。AWSなどの大手クラウドを含まないこの集計では、2026年5月4日から7月27日に、H100・H200の両方が上昇しました。単位は1基・1時間あたりの米ドルです。<a class="citation" href="#ref-44">[44]</a>

| GPU | 5月4日 | 7月27日 | 上昇率 |
| --- | ---: | ---: | ---: |
| H100 | 2.57 | 2.75 | 約7.0% |
| H200 | 2.70 | 3.09 | 約14.4% |

Blackwell世代のB200の指数も、2026年9月1日に5.62米ドルとなり、年初比で27.6%上昇しています。<a class="citation" href="#ref-45">[45]</a> 新しいGPUにも値上がりの例はありますが、この期間に何倍にもなったわけではありません。これらは見積もり料金に基づく指数で、全利用者の支払額を示すものではなく、B200は上の表と集計期間も異なります。利用料の動きから、GPU本体の販売価格も同じように動いたとはいえません。

### D.3 RTX 5090の掲載価格とメモリの供給制約

本文の小売価格は、Inno3D GeForce RTX 5090 X3 OC（型番：`GD5090-32GERX3OC`）の価格.com掲載最安価格です。同じ型番を追っていますが、途中で掲載店は変わり得ます。安い店の在庫がなくなり、高い店だけが残っても、この指標は上がります。<a class="citation" href="#ref-25">[25]</a>

| 日付 | 国内の掲載最安価格（税込） |
| --- | ---: |
| 2025年3月25日 | 460,000円 |
| 2025年8月1日 | 380,000円 |
| 2026年2月1日 | 659,800円 |
| 2026年4月1日 | 589,900円 |
| 2026年6月1日 | 669,800円 |
| 2026年9月17日 | 948,700円 |
| 2026年10月13日 | 1,129,800円 |

RTX 5090のメモリはHBMではなく、32GBのGDDR7です。<a class="citation" href="#ref-46">[46]</a> TrendForceは2026年9月の調査で、先端DRAMの生産能力がサーバー向けに振り向けられ、GDDR7の生産量が制限されていると説明しています。GDDR7自体もAIチップ向けの需要が強いとしています。<a class="citation" href="#ref-47">[47]</a> AI需要の影響はHBMだけでなく、ほかのメモリへの生産配分や需要にも及んでいます。ただし、こうした制約から個別製品の値上がり幅を算定することはできず、小売価格の上昇分をそのままNVIDIAの値上げや流通業者の利益と見なすこともできません。

<a id="appendix-compute-buyers"></a>

## Appendix E. 計算資源の調達規模をどう数えるか

OpenAIは2025年7月、Oracleと追加4.5GW分のデータセンター開発に合意しました。GWは設備規模を電力で表したもので、GPUの台数ではなく、発表時点で全量が稼働していたわけでもありません。<a class="citation" href="#ref-48">[48]</a> また、GPUを調達するクラウド事業者と、その計算能力を使う企業は別です。両者の発表を独立した需要として足すと、同じ設備を二重に数える可能性があります。

ほかの需要源として、Metaも2026年2月、数百万基のBlackwell・Rubin GPUの導入を含む複数年の提携を発表しています。<a class="citation" href="#ref-49">[49]</a> 一方、Anthropicは同年4月、Amazon独自のAIチップであるTrainium2を100万基超利用していると説明しています。<a class="citation" href="#ref-50">[50]</a> AI向け計算資源への需要を、すべてNVIDIA GPUへの需要として数えることはできません。供給に余裕が少なければ一部の企業の追加需要だけでも不足は起こり得ますが、これらの発表だけから両社の寄与率や、競合を締め出すために不要な設備を囲い込む意図までは確認できません。


<a id="appendix-alternative-software"></a>

## Appendix F. 代替チップを使うソフトウェアの役割

モデルの計算を記述するフレームワークと、それを各チップで実行するコンパイラ・ライブラリは別の役割です。例えばPyTorchはTritonを使うコード生成でNVIDIAとAMDに対応し、TPUにはXLAコンパイラを使うPyTorch/XLAという経路があります。<a class="citation" href="#ref-51">[51]</a><a class="citation" href="#ref-52">[52]</a> TPUはJAXからも利用でき、AWSのNeuronもPyTorch・JAXとの連携、コンパイラ、実行環境などを提供しています。<a class="citation" href="#ref-4">[4]</a><a class="citation" href="#ref-5">[5]</a> モデルを記述する方法を残しつつ、その下の実装を変えることは、移行時に書き直す範囲を減らす一つの方法です。ただし、使える機能や学習・推論の対応範囲は、ソフトウェアの版や実行経路にも依存します。

手書きのkernelには、さらにハードウェア固有の事情があります。Tritonの公式リポジトリが対応ハードウェアとして挙げるのはNVIDIAとAMDのGPUで、TPUやTrainiumでそのまま動くという意味ではありません。<a class="citation" href="#ref-53">[53]</a> Helionは、それより高い水準でkernelを記述し、GPU向けにはTriton、TPU向けにはPallasという言語へ変換する方向で共用を進めています。<a class="citation" href="#ref-54">[54]</a><a class="citation" href="#ref-11">[11]</a> これは任意のPyTorchモデル全体を自動的にPallasへ変換する機能ではありません。一方、AWSの移植agentは既存kernelをNKIへ書き換える方式であり、共通のコンパイラによる変換とは異なります。同社の150課題の評価では約83%がコンパイルに成功しましたが、数値の一致を検証できたのは別途57課題の部分集合で、そのうち50課題が判定を通過しています。コンパイル成功率を、そのまま移植完了率と読むことはできません。<a class="citation" href="#ref-14">[14]</a>

AMD向けにも、共通基盤を使える状態へ近づける開発の蓄積があります。ROCmへの対応にはMetaの開発者も協力し、PyTorch 1.12で安定版に移行しました。<a class="citation" href="#ref-7">[7]</a> さらにGeorge Hotz（geohot）らのtinygradは、計算グラフのコンパイラからGPUドライバまで独自に実装し、Hotzは2025年末にMI350Xでの大規模モデル学習に向けたAMDとの開発契約を報告しています。<a class="citation" href="#ref-55">[55]</a> RDNA3/4向けの独自AMドライバはGPUの初期化や仮想メモリ管理まで担い、既存のLinuxドライバを使わない経路も用意しています。<a class="citation" href="#ref-56">[56]</a><a class="citation" href="#ref-57">[57]</a> 演算コード以外にも、代替を使うための開発対象があるわけです。

AIによる開発支援も、そうした低レイヤーに及んでいます。2026年3月にはROCmの公開リポジトリに、Claudeを共著者として記録したPython製GPUドライバのコミットが公開されています。Linux側のドライバは利用するものの、メモリ転送や計算命令の構築まで含む実装です。<a class="citation" href="#ref-58">[58]</a> ただし、この記録だけからAIが担当した範囲や開発費用の削減率まではわかりません。また、NVIDIA向けの開発にもAIは使えるため、開発の効率化が最終的にどの企業を有利にするかは別の問題です。

<!-- build:ignore:start -->
## 執筆メモ：本文全体で区別すること

供給が足りない理由と、NVIDIAが利益を得られる理由を混同しない。HBMなどの希少性による利益が、サプライチェーンのどの企業に帰属するかは別の問題。全社の売上総利益率をGPU単体の利益率や限界費用に基づく利幅として使わない。GPU単体、ボード、サーバー、ラック、クラウド利用料の価格を同じものとして比較しない。製品世代や販売条件を特定したうえで一次資料を確認する。

## 次の記事にいれる

仮題は「HBMを小さなダイに分ければ、もっと安く作れるのか」。今回の記事から分ける中心の話題は、DRAMダイそのものをさらに細分化して組み合わせる設計に改善の余地があるか、という問い。単にHBMの積層数を増やす話に置き換えない。

### 出発点と、混同しない三つの変更

チップレットという考え方を計算回路だけでなくメモリにも適用できるかを確認する。そのうえで、既存のDRAMダイの積層数を増やすこと、完成したHBMスタックを横に増やすこと、DRAMダイ自体をさらに小さく分けることを区別する。既存のHBMが複数ダイを積んでいるという説明だけでは、最後の提案への回答にならない。

### 小さなダイに分ける利点と追加負担

小さく分けることで、製造欠陥があったときに捨てる面積を減らせるか、良品ダイを選別して組み立てられるかを調べる。対して、各ダイに必要な周辺回路、入出力、TSV、配線、切り出し領域、検査・接合回数がどれだけ増えるかを比較する。ダイの歩留まりと積層・接合の歩留まりは別に扱い、良品選別を無視して単純にダイ歩留まりを積層数だけ掛ける計算にしない。DRAMの冗長セルや修復の仕組みも、微細化したダイの利点を評価する際の調査項目とする。

### 容量、帯域幅、実装条件をそろえて比較する

容量を増やすことと帯域幅を増やすことは別の設計変更として扱う。同容量の構成同士で、外部インターフェイス、GPU側のメモリコントローラ、配線密度、消費電力、レイテンシを確認する。縦に積む場合は高さ、薄化、熱、接合を、横に並べる場合はパッケージ面積と配線距離を調べる。DDR5用の完成ダイをそのまま集めればHBMになる、という前提は置かない。

### 何を改善できたら「安くなる」といえるか

主要な比較指標は、投入ウェハー面積あたりの完成良品容量と、接合・パッケージング・検査を含む完成良品1GBあたりの総コスト。帯域幅などの要求を満たすことを比較条件にする。前工程の効率を改善しても後工程が新たなボトルネックになる可能性を考える。メモリ部品の共通化・再利用によって需要変動へ対応しやすくなるかも、製造歩留まりの改善とは別の利点として検討する。

小さな数値モデルを作るなら、同容量の大きなダイ数個と小さなダイ多数を比較する。ダイ面積、周辺回路の増分、欠陥と修復、良品選別、接合歩留まり、組み立て費用を明示する。実データで裏づけられない値は説明用の仮定とし、現在のHBMのダイサイズが最適かどうかは結論を先取りしない。

## 保留：設備の共通化と供給過剰

HBMと通常DRAMの生産配分はAppendix Aに記載済み。さらに掘り下げる場合は、同一プロセスでGPUとCPUを作る場合や、HBMと通常DRAMの前工程の生産配分を変える場合について、共通化できる範囲と専用工程を調べる。独立して長くなるなら別記事に分ける。製造途中の製品を変換することと、新規に投入する生産枠を変更することを区別する。

設備を転用できても、転用先の需要が増えるとは限らない。各社が同時にDDR5へ切り替えると、供給増加と価格下落によって採算が悪化する可能性を数量競争の仮想例で説明する。需要の相関、専用設備の効率と汎用設備の柔軟性、固定費と操業判断、長期購入契約によるリスク分担も関連する。企業の利益が下がることと、消費者を含む社会全体の損失は同一視しない。

## 調査の出発点

- [元の共有会話](https://chatgpt.com/share/6accf470-7718-83e8-953d-bcf3e1c7a53c)。論点の記録として使い、技術的事実の根拠は一次資料で取り直す。
- [Micronの2026会計年度年次報告書](https://www.sec.gov/Archives/edgar/data/723125/000072312526000023/mu-20260903.htm)。同一製造世代・同ビット数あたりのHBMと通常DRAMの製造資源の違い、生産能力の転用、設備増設の計画を確認できる。共有会話にあったHBM3Eの具体的な倍率は、旧決算資料のURLが本文に到達しなかったため今回の記事には採用していない。次回使う場合は閲覧できる一次資料と対象条件を確認する。
- [NVIDIAの年次報告書一覧](https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx)。供給能力、競争、売上原価・利益率の定義を確認する。
- 次回記事では、メモリメーカーのHBM技術資料、積層メモリの歩留まり・コストを分析した原論文、AMD等のメモリ構成に関する公式資料を調べる。共有会話に出てきた製品容量・積層数や将来予測を、そのまま確定事項として転載しない。

<!-- build:ignore:end -->

## 参考文献

<ol class="references">
  <li id="ref-1">NVIDIA Corporation. <cite><a href="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm">Form 10-Q for the quarterly period ended July 26, 2026</a></cite>. 2027会計年度第2四半期。Item 2「Management's Discussion and Analysis of Financial Condition and Results of Operations」、特に「Gross Profit and Gross Margin」。</li>
  <li id="ref-2">NVIDIA Corporation. <cite><a href="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm">Form 10-K for the fiscal year ended January 25, 2026</a></cite>. 2026会計年度。Item 1「Business」の「Data Center」「Manufacturing」「Competition」、Item 7「Concentration of Revenue」。直接・間接の販売先と調達経路についても参照。</li>
  <li id="ref-3">Advanced Micro Devices, Inc. <cite><a href="https://rocmdocs.amd.com/projects/HIPIFY/en/latest/index.html">HIPIFY documentation</a></cite>. CUDAからHIPへの変換の範囲と、変換後の検証・性能調整について。2026年10月13日閲覧。</li>
  <li id="ref-4">Google Cloud. <cite><a href="https://docs.cloud.google.com/tpu/docs/intro-to-tpu">Introduction to Cloud TPU</a></cite>. TPUの用途、XLAコンパイラの役割、JAX・PyTorchからの利用について。2026年10月13日閲覧。</li>
  <li id="ref-5">Amazon Web Services. <cite><a href="https://aws.amazon.com/ai/machine-learning/neuron/">AWS Neuron</a></cite>. Trainium・Inferentia向けの開発基盤。コンパイラ、実行環境、PyTorch・JAXとの連携について。2026年10月13日閲覧。</li>
  <li id="ref-6">Anthropic. <cite><a href="https://www.anthropic.com/news/google-broadcom-partnership-compute">Anthropic expands partnership with Google and Broadcom for multiple gigawatts of next-generation compute</a></cite>. 2026年4月6日。Claudeの学習・実行にTPU、Trainium、NVIDIA GPUを利用していることについて。</li>
  <li id="ref-7">AMD. <cite><a href="https://pytorch.org/blog/democratizing-ai-with-pytorch/">Democratizing AI with PyTorch Foundation and ROCm™ support for PyTorch</a></cite>. PyTorch Blog, 2023年2月14日。Metaの開発者を含む共同開発と、PyTorch 1.12でのROCm対応の安定版への移行について。</li>
  <li id="ref-8">Advanced Micro Devices, Inc. <cite><a href="https://newsroom.amd.com/news/meta-ai-infrastructure-update/">From EPYC to Helios, AMD and Meta are Scaling the Future of AI Together</a></cite>. 2026年7月23日。「Co-Engineering AI Infrastructure with Instinct GPUs」。MetaとのROCmの最適化、推薦・ランキングモデルの学習への協力拡大について。</li>
  <li id="ref-9">Advanced Micro Devices, Inc. <cite><a href="https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era">AAI 2026: AMD Delivers Full-Stack Compute for the Agentic AI Era</a></cite>. 2026年7月23日。OpenAIとのTriton・ROCmを用いたAMD Instinct MI455X向け最適化について。</li>
  <li id="ref-10">Anthropic. <cite><a href="https://www.anthropic.com/news/anthropic-amazon-trainium">Powering the next generation of AI development with AWS</a></cite>. 2024年11月22日。「Collaboration on AWS Trainium hardware and software」。低レベルkernelの実装、Neuronへの貢献、将来世代のTrainiumの開発への協力について。</li>
  <li id="ref-11">Dunfan Luほか. <cite><a href="https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/">Helion on TPU: Towards Hardware Heterogeneous Kernel Authoring</a></cite>. PyTorch Blog, 2026年7月23日。Googleとの共同開発によるPallasコード生成と、TPU向けkernelの最適化について。</li>
  <li id="ref-12">Gang Liaoほか, Meta. <cite><a href="https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/">KernelEvolve: How Meta’s Ranking Engineer Agent Optimizes AI Infrastructure</a></cite>. 2026年4月2日。複数のハードウェアを対象とするkernel生成・最適化と、開発時間短縮の報告。</li>
  <li id="ref-13">Advanced Micro Devices, Inc. <cite><a href="https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html">GEAK v4: From Faster Kernels to Faster Workloads</a></cite>. 2026年7月23日。「GEAK v4 Architecture」「Evaluation」。HIP・Tritonなどのコードの生成・最適化と実機での検証について。</li>
  <li id="ref-14">Junjie Tang, Daryl Hammett, Tushar Saxena, Amazon Web Services. <cite><a href="https://aws.amazon.com/blogs/migration-and-modernization/stop-hand-porting-kernels-migrate-pytorch-and-triton-kernels-to-aws-trainium-with-an-aws-transform-custom-agent/">Stop hand-porting kernels: migrate PyTorch and Triton kernels to AWS Trainium with an AWS Transform custom agent</a></cite>. 2026年7月30日。NKIへの移植と実機での検証。「Results」の150課題のコンパイル結果と57課題の数値検証結果。</li>
  <li id="ref-15">Sasha Krassovsky. <cite><a href="https://blog.krassovsky.com/">Sasha’s Site</a></cite>. 「Bio」「Projects」。Anthropicへの所属と、Metal kernelを記述するSporkの紹介。2026年10月13日閲覧。</li>
  <li id="ref-16">Sasha Krassovsky. <cite><a href="https://github.com/save-buffer/spork/commit/4a262329db702e1ebb6f67daab2fbf2948c736a4">spork.kernels library + JittedKernel.bind()</a></cite>. save-buffer/spork, commit 4a262329db702e1ebb6f67daab2fbf2948c736a4, 2026年5月18日（UTC）。行列積・行列加算kernelの追加と、Claude Opus 4.7の共著者記録。</li>
  <li id="ref-17">Sasha Krassovsky. <cite><a href="https://github.com/save-buffer/spork/commit/c70dc83258d498b9f8b8c2a7f0b333f2b8d949e5">sk.kernels.causal_gqa: fused FA-2 + MPP</a></cite>. save-buffer/spork, commit c70dc83258d498b9f8b8c2a7f0b333f2b8d949e5, 2026年5月18日（UTC）。attention kernelの追加と、Claude Opus 4.7の共著者記録。</li>
  <li id="ref-18">Sasha Krassovsky. <cite><a href="https://github.com/save-buffer/spork/commit/ca9c909e4df8d1e41922500edd31a7a7e885d751">Fix causal_gqa: four bugs found by running the test</a></cite>. save-buffer/spork, commit ca9c909e4df8d1e41922500edd31a7a7e885d751, 2026年5月21日（UTC）。attention実装の修正と、Claude Opus 4.7の共著者記録。</li>
  <li id="ref-19">Taiwan Semiconductor Manufacturing Company Limited. <cite><a href="https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2026-08/3e494f0c14dd0890f897aa044415e21d93486cc4/TSMC%202Q26%20Transcript.pdf">Q2 2026 Earnings Call Transcript</a></cite>. 2026年7月16日。LSEG編集、TSMC公開。p. 4、2026年の設備投資予算の引き上げについて。</li>
  <li id="ref-20">ASML. <cite><a href="https://ourbrand.asml.com/asset/1fd3908a-0381-47b5-9b69-a3094f656651/2026_07_15-ASML-Transcript-investor-call-Q2-2026.pdf">Transcript ASML Investor Call, Q2 2026</a></cite>. 2026年7月15日。pp. 2–3、顧客による先端ロジック・DRAMの能力増強と、長期契約による需要の見通しについて。</li>
  <li id="ref-21">Advanced Micro Devices, Inc. <cite><a href="https://ir.amd.com/financial-information/sec-filings/content/0000002488-26-000018/amd-20251227.htm">Form 10-K for the fiscal year ended December 27, 2025</a></cite>. 2026年2月4日提出。Item 1「Manufacturing Arrangements and Assembly and Test Facilities」、Item 1A「Operational and Technology Risks」。</li>
  <li id="ref-22">Micron Technology, Inc. <cite><a href="https://www.sec.gov/Archives/edgar/data/723125/000072312526000023/mu-20260903.htm">Form 10-K for the fiscal year ended September 3, 2026</a></cite>. 2026会計年度。Item 1A「Risk Factors」の製品開発・製造・設備拡張・長期顧客契約に関するリスク、Item 7「Liquidity and Capital Resources」の設備増設計画。</li>
  <li id="ref-23">Micron Technology, Inc. <cite><a href="https://micron.gcs-web.com/static-files/2354ecda-77a0-4ddd-8462-a631eb491356">Fiscal Q3 2026 Earnings Call</a></cite>. 2026年6月24日。決算説明スライド、p. 6「Industry Trends (2 of 3)」。製造世代の移行によるビット生産量の伸びの鈍化と、工場新設の必要性についての同社の見解。</li>
  <li id="ref-24">Micron Technology, Inc. <cite><a href="https://s25.q4cdn.com/621799436/files/doc_financials/2026/q4/Q4-FY26-Prepared-Remarks.pdf">Fiscal Q4 2026 Earnings Call Prepared Remarks</a></cite>. 2026年9月30日。pp. 2–3「Manufacturing expansions」、p. 3「Strategic customer agreements update」、p. 5「Market outlook」。Tongluoの出荷計画、新工場の生産量の立ち上げ、長期顧客契約と2027会計年度の設備投資計画について。</li>
  <li id="ref-25">価格.com. <cite><a href="https://kakaku.com/item/K0001679253/pricehistory/">Inno3D GeForce RTX 5090 X3 OC GD5090-32GERX3OC [PCIExp 32GB] 価格推移グラフ</a></cite>. グラフの各日の掲載最安価格を参照。Amazonは同サイトの履歴集計から除外されている。2026年10月13日閲覧。</li>
  <li id="ref-26">Nebius team. <cite><a href="https://nebius.com/blog/posts/digest-august-2024">Nebius monthly digest, August 2024</a></cite>. 2024年9月6日。「Price drop: Get NVIDIA H100 GPUs for less」。4.85米ドルから3.50米ドルへの値下げ報告。</li>
  <li id="ref-27">Nebius. <cite><a href="https://nebius.com/prices">AI Cloud pricing</a></cite>. 「NVIDIA GPU Instances」。H100の通常料金を参照。2026年10月13日閲覧。</li>
  <li id="ref-28">Nebius team. <cite><a href="https://nebius.com/blog/posts/early-access-unlocked">Early access to Nebius AI is officially unlocked</a></cite>. 2023年11月1日。H100 SXM5の提供開始時の従量課金。</li>
  <li id="ref-29">Jeffrey Kampman, Paul Alcorn, Tom's Hardware. <cite><a href="https://www.tomshardware.com/pc-components/gpus/gigabyte-ceo-explains-nvidias-potential-gpu-supply-strategy-amid-crushing-memory-shortages-gross-revenue-per-gigabyte-of-gddr7-memory-could-decide-what-products-thrive">Gigabyte CEO explains Nvidia's potential GPU supply strategy amid crushing memory shortages</a></cite>. 2026年1月16日。記事中のNVIDIAによる取材への回答を参照。GeForce RTXの需要とメモリ供給の制約について。</li>
  <li id="ref-30">石橋一衛, AKIBA PC Hotline!. <cite><a href="https://akiba-pc.watch.impress.co.jp/docs/news/news/2078554.html">上位のGeForce RTX 50シリーズの品薄続く。価格表は真っ白、在庫切れでショーケースを布で隠す店も</a></cite>. 2026年1月17日。秋葉原の販売店への取材。</li>
  <li id="ref-31">Amazon Web Services. <cite><a href="https://aws.amazon.com/ec2/capacityblocks/pricing/">Amazon EC2 Capacity Blocks for ML pricing</a></cite>. 「Pricing tables」。需給に応じた予約料金の更新と購入済み予約への適用条件。2026年10月13日閲覧。</li>
  <li id="ref-32">Anthropic. <cite><a href="https://www.anthropic.com/news/microsoft-nvidia-anthropic-announce-strategic-partnerships">Microsoft, NVIDIA, and Anthropic announce strategic partnerships</a></cite>. 2025年11月18日。Azureの計算資源への購入コミットメントとNVIDIA GPUの利用計画について。</li>
  <li id="ref-33">Uksong Kang, SK hynix. <cite><a href="https://news.skhynix.com/en/creating-new-values-in-dram-using-through-silicon-via-technology-for-continued-scaling-in-memory-system-performance-and-capacity/">Creating New Values in DRAM Using Through-Silicon-Via Technology for Continued Scaling in Memory System Performance and Capacity</a></cite>. 2019年11月20日。TSVによる積層、TSVの面積と組み立て歩留まりの関係。</li>
  <li id="ref-34">Taiwan Semiconductor Manufacturing Company Limited. <cite><a href="https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm">CoWoS®</a></cite>. GPUなどのロジックとHBMを一つのパッケージに組み合わせる技術について。2026年10月13日閲覧。</li>
  <li id="ref-35">SK hynix. <cite><a href="https://news.skhynix.co.kr/hidden_nix_ep1/">[히든닉스 1편] 공정과 건축 사이, 팹 짓는 전문가들 ‘건설건축 팀’</a></cite>（工場を建設する建設・建築チームへのインタビュー）. 2025年9月5日。「Chapter 2. 건축과 공정을 이해한다」。設計と施工の並行化、M11のクリーンルーム稼働までの期間、M15Xでの工法の改良について。</li>
  <li id="ref-36">Taiwan Semiconductor Manufacturing Company Limited. <cite><a href="https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2023-07/7ec677062ca442e429b632ccd6d4f31ad53b1ce7/TSMC%202Q23%20Transcript.pdf">Q2 2023 Earnings Call Transcript</a></cite>. 2023年7月20日。Refinitiv編集、TSMC公開。p. 6、アリゾナ第1工場の装置設置に必要な熟練者の不足と、台湾からの技術者派遣について。</li>
  <li id="ref-37">Micron Technology, Inc. <cite><a href="https://investors.micron.com/news/press-release/2026/Micron-Completes-Acquisition-of-PSMCs-Tongluo-P5-Site-in-Taiwan-03-15-2026/default.aspx">Micron Completes Acquisition of PSMC’s Tongluo P5 Site in Taiwan</a></cite>. 2026年3月15日。既存クリーンルームの取得と改修計画について。</li>
  <li id="ref-38">Micron Technology, Inc. <cite><a href="https://micron.gcs-web.com/news-releases/news-release-details/micron-announces-shipment-1g-1-gamma-dram-pioneering-memory">Micron Announces Shipment of 1γ (1-gamma) DRAM: Pioneering Memory Technology Advancements for Future Compute Needs</a></cite>. 2025年2月25日。「Improved bit-density output」と注3。1βと1γのウェハーあたりのビット数の比較。</li>
  <li id="ref-39">Taiwan Semiconductor Manufacturing Company Limited. <cite><a href="https://www.tsmc.com/english/dedicatedFoundry/technology/platform_HPC_tech_advancedTech">HPC Platform – Advanced Technologies</a></cite>. 「N7/N6」。EUVによる工程の簡略化、製造時間の短縮、歩留まりの改善について。2026年10月13日閲覧。</li>
  <li id="ref-40">SK hynix. <cite><a href="https://news.skhynix.com/en/meet-the-sk-hynix-team-behind-the-worlds-first-12-layer-hbm3/">Strengthening HBM Market Leadership: Meet the SK hynix Team Behind the World’s First 12-Layer HBM3</a></cite>. 2023年5月19日。MR-MUFの仕組み、フィルム状材料を使う方式との比較、12層HBM3への改良版の適用について。</li>
  <li id="ref-41">Channy Yun, Amazon Web Services. <cite><a href="https://aws.amazon.com/blogs/aws/announcing-up-to-45-price-reduction-for-amazon-ec2-nvidia-gpu-accelerated-instances/">Announcing up to 45% price reduction for Amazon EC2 NVIDIA GPU-accelerated instances</a></cite>. 2025年6月5日。5月31日を基準とするP4d・P4de・P5の値下げ率。オンデマンド料金への適用は6月1日から。</li>
  <li id="ref-42">Chuan Li, Lambda. <cite><a href="https://lambda.ai/blog/how-to-use-mpirun-to-launch-a-llama-inference-job-across-multiple-cloud-instances">How To Use mpirun to Launch a LLaMA Inference Job Across Multiple Cloud Instances</a></cite>. 2023年3月14日公開、同年8月3日更新。「Cost」のA100 40GB搭載インスタンスの料金。</li>
  <li id="ref-43">Lambda. <cite><a href="https://lambda.ai/pricing">AI cloud pricing</a></cite>. 「Instances pricing」。1基構成のA100 40GBの通常料金。2026年10月13日閲覧。</li>
  <li id="ref-44">Carmen Li, Silicon Data. <cite><a href="https://www.silicondata.com/blog/h200-vs-h100-rental-prices-may-july-2026">H200 vs H100 Rental Prices, May to July 2026: The Premium That Doubled</a></cite>. 2026年8月4日。Neo-Cloud指数の2026年5月4日・7月27日の値と集計対象について。</li>
  <li id="ref-45">Silicon Data. <cite><a href="https://www.silicondata.com/blog/b200-rental-price-august-2026-update">B200 Rental Price Update, August 2026: The Quietest Month Yet</a></cite>. 2026年9月24日付。2026年9月1日のB200指数と年初比の変化、見積もり料金を用いる指数の定義を参照。</li>
  <li id="ref-46">NVIDIA. <cite><a href="https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/">GeForce RTX 5090</a></cite>. 「Specs」のメモリ仕様。2026年10月13日閲覧。</li>
  <li id="ref-47">TrendForce. <cite><a href="https://www.trendforce.com.tw/presscenter/news/20260930-13256.html">AI Server增量需求支撐4Q26記憶體合約價漲勢，消費端壓力未減</a></cite>. 2026年9月30日。Graphics DRAMの調査。GDDR7のAI向け需要と、サーバー向けDRAMへの生産能力配分による供給制約について。</li>
  <li id="ref-48">OpenAI. <cite><a href="https://openai.com/index/stargate-advances-with-partnership-with-oracle/">Stargate advances with 4.5 GW partnership with Oracle</a></cite>. 2025年7月22日。追加のデータセンター開発に関する合意と、当時の建設・稼働状況について。</li>
  <li id="ref-49">NVIDIA. <cite><a href="https://nvidianews.nvidia.com/news/meta-builds-ai-infrastructure-with-nvidia">Meta Builds AI Infrastructure With NVIDIA</a></cite>. 2026年2月17日。MetaによるBlackwell・Rubin GPUの導入を含む複数年の提携について。</li>
  <li id="ref-50">Anthropic. <cite><a href="https://www.anthropic.com/news/anthropic-amazon-compute">Anthropic and Amazon expand collaboration for up to 5 gigawatts of new compute</a></cite>. 2026年4月20日公開、同21日更新。既に利用しているTrainium2の規模について。</li>
  <li id="ref-51">PyTorch. <cite><a href="https://pytorch.org/get-started/pytorch-2-x/">PyTorch 2.x</a></cite>. TorchInductorによるTritonコード生成とNVIDIA・AMD GPUへの対応について。2026年10月13日閲覧。</li>
  <li id="ref-52">PyTorch/XLA contributors. <cite><a href="https://docs.pytorch.org/xla/master/learn/eager.html">PyTorch/XLA Compile API and it’s interaction with Eager mode.</a></cite>. 「torch_xla.compile vs. torch.compile」。XLAを使う実行経路と、学習・推論におけるコンパイルAPIの違いについて。2026年10月13日閲覧。</li>
  <li id="ref-53">Triton contributors. <cite><a href="https://github.com/triton-lang/triton">Triton language and compiler</a></cite>. README「Compatibility」の対応ハードウェア。2026年10月13日閲覧。</li>
  <li id="ref-54">PyTorch Helion contributors. <cite><a href="https://github.com/pytorch/helion">Helion</a></cite>. README「About」「Example」。Tritonへのコード生成と自動チューニングについて。2026年10月13日閲覧。</li>
  <li id="ref-55">George Hotz. <cite><a href="https://geohot.github.io/blog/jekyll/update/2025/12/29/five-years-of-tinygrad.html">Five years of tinygrad</a></cite>. 2025年12月29日。独自ソフトウェアの実装範囲と、MI350XでLlama 405Bの学習をMLPerfに載せるためのAMDとの契約についての本人の報告。</li>
  <li id="ref-56">tinygrad. <cite><a href="https://docs.tinygrad.org/developer/am/">AM Driver</a></cite>. RDNA3/4向けドライバのGPU初期化、キュー設定、仮想メモリ管理について。2026年10月13日閲覧。</li>
  <li id="ref-57">tinygrad. <cite><a href="https://docs.tinygrad.org/runtime/">Runtimes</a></cite>. 「AMD Interfaces」。既存のamdgpuドライバを利用するKFD経路と、独自AMドライバを使うPCI経路について。2026年10月13日閲覧。</li>
  <li id="ref-58">powderluv, Claude（共著者表記）. <cite><a href="https://github.com/ROCm/TheRock/commit/ba0c4625751ffa2f7a1d3679421380efa4e13478">Add pure-Python AMD GPU userspace driver</a></cite>. ROCm/TheRock, commit ba0c4625751ffa2f7a1d3679421380efa4e13478, 2026年3月2日。LinuxのKFDを利用するドライバの実装と、Claude Opus 4.6の共著者記録。</li>
</ol>
