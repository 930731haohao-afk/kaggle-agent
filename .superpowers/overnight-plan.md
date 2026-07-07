# 過夜自主計畫:跨季泛化首航(2026-07-06 晚 → 07-07 早)

使用者已授權:「開跑+提交」。前置:等報告調整批次 1–3 完成後啟動。
**每完成一單元即更新進度區 + commit。**

## 鐵則(承襲週末計畫,全程適用)
- 一律 `uv run`;工作目錄 `/home/tjyen/ai_agents/kaggle`;訓練循序不平行
- 實驗紀錄一律 `log_experiment_v2`;報告遵守新規範(R-W1~W7 + 工具/效能對照/總結三新節)
- **提交已授權**:每場最終最佳解提交 1 次(Late Submission 開放者;token 在 ~/.kaggle/kaggle_api_token.txt;提交前必過 OOF 重現閘門,比照 s3e16 流程);403/關閉則記錄跳過
- 對話 compact 後:讀本檔進度區 + git log,從第一個未完成單元續跑,絕不重跑
- 被權限 deny 的指令跳過並記錄

## 對象:5 場跨季 Playground(資料已在磁碟;各場先讀既有 config.yaml/experiments.json/STATUS.md,二月的舊紀錄如實保留,tier 歸屬誠實判定)
s4e1 → s4e11 → s5e10 → s6e1 → s6e2(小→大/舊→新;s6e2 有既有訓練紀錄,續用不重跑)

## 每場單元(×5)
1. tier1:generic 基線(run_competition.py;若二月已有等效 generic 紀錄則沿用,不重跑)
2. tier2:kaggle-agent skill 完整六階段(subagent 讀 SKILL.md;EDA→特徵→建模→submission 檔)
3. tier3:self-improvement 迭代(經驗庫先驗;2–4 輪;連 2 輪無改善停)
4. tier4:樹搜尋 v3(harness_v3 預算策略;OOF cache;H-1 resume/timeout/sanity gate 首次無人實戰)
5. 報告:先 `uv run python3 .claude/skills/kaggle-report/assets/eda_summary.py <comp>`(四來源之 EDA 接入,2026-07-06 已整合)→ collect.py → 新格式 7 節(含四層效能對照);verify exit 0;<short>_REPORT.pdf
6. 提交:重建最終最佳 test 預測(OOF 閘門)→ kaggle submit → 抓 LB 分數寫回 experiments.json/報告;關閉則標 CV-only
7. commit(訊息 `feat(<comp>): 跨季泛化全流程+LB`)

## 收尾單元
- benchmark facts/summary 擴到 15 場(新增「跨季泛化(S4/S5/S6)」章節;經驗庫是否跨季成立的誠實評估)
- SUMMARY_REPORT.md/pdf 同步更新
- 經驗庫新增跨季驗證條目(成立與不成立都記)
- overnight_summary 簡短總結(併入 SUMMARY_REPORT 或獨立小節)
- 記憶檔更新

## 進度區
- [x] 計畫建立(2026-07-06 16:10)
- [ ] 前置:報告調整批次 1(規範+前5場)/ 批次 2(後5場)/ 批次 3(SUMMARY_REPORT)

## 併行 session 協定(2026-07-06 16:50 使用者指示:與 VS Code session 同時跑)
每次 commit 前必做:
1. 只用明確路徑 git add(禁 -A/.);staged 清單必須 == 本單元自己產的檔案
2. `git status --short` 檢查:若發現非本單元的變更(他 session 的),一律不 staged、不覆蓋、不還原
3. 若本單元要改的檔案出現他 session 的未預期修改 → 以磁碟現狀為準重新驗證(verify/tests)後才 commit
4. index.lock 存在 → 等 3 秒重試,最多 5 次
5. commit 後 `git show --stat` 確認只含預期檔案

## Phase J(過夜任務全部收尾後啟動):樹搜尋優化——外部文獻想法注入(使用者 2026-07-06 17:55 交付;ERA 第二支柱/計畫書 4.1)
J-1 建外部想法庫 knowledge/idea_bank.md:WebSearch/WebFetch 蒐集表格型競賽的高可信技巧來源
    (Kaggle 冠軍 write-up、高引用論文如 target encoding/GBDT/stacking 文獻、教科書級技巧),
    每條:出處引文|機制|適用條件|預期成本|與現有經驗庫的關係([EXT] 標記)。≥20 條,來源必須真實可查,禁止捏造引文。
J-2 harness v4 注入鉤:suggest_priors 合併 idea_bank([EXT])與 experience.md([INT]),
    出處標記進 mutation 記錄;加「重組(recombination)」mutation 型(取兩個既有高分節點的核心概念由 agent 合成新解,計畫書 4.1)。+測試,全綠。
J-3 tier5 定義(使用者 2026-07-06 18:00 指示):樹搜尋+想法注入 = **tier5**,正式入消融對照。
    - 控制變因:tier5 跑必須與 tier4 同 folds/同評估器,唯一差異 = idea_bank 注入(+重組 mutation),v4 vs v3 乾淨對照
    - 結果以 log_experiment_v2 入帳(notes 引 idea_bank 條目出處 + tree node)
    - 先 2-3 場深度歸因跑([EXT] vs [INT] vs uninformed 勝率),再依時間掃其餘場次;未跑到的場次 tier5 標「未執行」
J-4 schema 全面升級:report_structure §5 效能對照 五層 schema(tier5 選填列);
    build_benchmark_table.py + benchmark_facts/summary + SUMMARY_REPORT 加 tier5 欄;各場已有 tier5 者報告補列。
J-5 更新 tree_search_prototype 報告(想法注入章+五層結論)+ 經驗庫 + 記憶。
- [x] 批次 1+2+四來源整合完成(595e83b/7f2c6c9/5ad5191;10/10 場新格式,verify 全過)
- [!] 2026-07-07 09:55 檢討:昨夜批次2主控卡死子agent等待整夜,喚醒鏈中斷→跨季5場未跑。補救:批次3與跨季並行啟動;新增規則:主控等待子agent逾40分無進展即自行序列執行,不再無限等待
- [x] 批次 3 完成(commit 14b9c9e;docs/SUMMARY_REPORT.md+pdf,verify 過)——報告改造任務全數收工
- [x] 跨季單元 2/5 s4e11 完成(四階段 0.93955→0.939488→0.9399→0.940235,階段2小退如實記;3.1 經驗庫先驗 is_unbalance 對照檢驗否決——負結果入經驗庫;階段4 樹搜尋 60/60 滿預算,node#26 8成員 blend 勝 mega-blend 0.940028 與爆發長射(與「mega-blend 常勝」經驗相反,已記回經驗庫),4.4 後備種子入最佳解、4.5 相位機首次跨季實戰;OOF 閘門逐位一致,門檻 0.49;提交/查詢皆 403→CV-only(二月錨點 Pub 0.94093/Priv 0.93939);執行 agent 13:18 遭誤停、主控 15:1x 依 40 分規則接手,完成 v4 報告(verify=0,PDF 6 頁含目錄頁碼)與 commit)
- [x] 跨季單元 1/5 s4e1 完成(四層 0.8922→0.894302→0.894354→0.894393;tier4 樹搜尋 60 節點滿預算,node#44 37成員 mega-blend 勝 tier3;H-1 load_search_state 首次真實復原成功(前代被 kill 於播種中,5 節點逐位元載回+補種),並修復 driver「lineage 提案耗盡空轉」缺口;OOF 重現閘門 37/37 全過 max|d|=0.0 → sub_tree_best_0.89439;LB 關閉實測 403(tier3/tier4 檔各一次)→ CV-only;exp17 入帳;REPORT 7 節含 R-W9 新 §2(工具/術語/刻度)verify=0 + s4e1_REPORT.pdf)

## 🆕 格式再迭代(2026-07-07 13:00 使用者指示):前言抽共同內容
- 新增 docs/PREFACE.md+pdf:全案共同的研究設計/流程/工具鏈/術語表/階段字典(含新「簡稱」欄)集中於此
- 各場報告 §2 瘦身為「2 本場工具、術語與採用策略」:只列本場特有工具(2.1)/本場特有術語(2.2)/本場採用情形精簡表 編號+簡稱+本場(2.3,定義不再重複);開頭一行指向《前言》
- s3e16 樣板已按此瘦身(verify=0,PDF 已重生),與 PREFACE 一起待使用者審查
- 鋪開 agent a102fcfad7207639d 已於 13:00 被我主動停止(避免再產 7 份立刻過時的全字典 §2);它已完成:規範3檔+s3e1/s3e3/s3e5/s3e7/s3e9 報告(階段語言,舊 §2 全字典)——這些之後只需 §2 瘦身
- 審查通過後:重派一個全新完整任務書的鋪開 agent(規範3檔改為前言指向式+11份報告 §2 瘦身與階段語言補齊+SUMMARY 對齊+一次 commit 含 PREFACE)
- s4e11 agent 照舊執行;其報告落地後補做 §2 瘦身即可
- [x] 2026-07-07 13:1x 目錄+頁碼入管線:md2pdf.sh 改 weasyprint 主引擎(chromium 備援,exit 0/2/3 契約不變),自動注入目錄(h2+h3)於第一個 h2 前;CSS @bottom-center 頁尾「N / 總頁數」+ .report-toc 樣式含 target-counter 各節頁碼;PREFACE/s3e16/SUMMARY 三份 PDF 已重生並 pdftotext 驗證(目錄+各節頁碼+頁尾頁碼全出)。使用者要求報告/前言/SUMMARY 都要目錄頁碼→已全面生效(所有 PDF 走同一管線)
- [x] 13:2x 使用者核可 v4 樣板(「目前這板不錯」)→ 鋪開 agent 已派出(A組 s3e1/e3/e5/e7/e9 只瘦身§2;B組 s3e11/e14/e19/e20/s4e1 全面對齊;+SUMMARY 階段語言+規範3檔;單一 commit,明確路徑 add,排除 .superpowers 與 s4e11 目錄)

## ⚡ CONTEXT-CLEAR 快照(2026-07-07 下午;/clear 後醒來先讀這段+git log)
**⟪2026-07-07 15:00 更新⟫ 格式 v4 已鋪開完成(commit 1bbe9b9,12 文件+規範全對齊;抽查過)**:權威=docs/PREFACE.md(前言:共同方法/工具/術語/階段字典)+ s3e16 REPORT.md(樣板:§2 三小節瘦身式)。四條使用者增補已全落地:表頭「定義/摘要」、二元欄「是否」句式、表內主階段粗體 **N**、「口徑」禁用。PDF 管線=weasyprint 主引擎,自動目錄(h2+h3)+頁尾頁碼。收尾中:前言 §5 已擴 6 詞(探針/第0折代理評估/OOF-only/巢狀驗證/rank-average/深耕時期),一個小 agent 正在把 10 場報告 §2.2 與前言重複的定義刪除+commit。
**以下為舊快照(結構描述已被 v4 取代,僅供追溯)**:
- 結構:1競賽目的/2使用工具、術語與階段定義(2.1工具/2.2術語表/2.3採用策略=完整字典表+本場未使用標記)/3實驗方法/4實驗軌跡/5效能對照(逐階段)/6總結/7重現指令
- 規則:階段1-5+子階段X.Y(唯一階段語言,tier/Phase禁用僅2.3留一行舊代號對照);R-W9先定義再使用;R-W10白話不自創詞、內部檔名/行話禁入正文(僅§7 code block可留腳本名);表頭nowrap;verify豁免\d.\d階段記號
- R-W10 追加例(2026-07-07):「消融」「邊際貢獻」「錨點」等學術詞禁用,一律白話(如「同一場解五次、一次多給一項方法、比較分數」)
- 表頭用詞(2026-07-07 13:5x 使用者):禁「一句話定義/一句話摘要」,一律「定義/摘要」;前言+樣板已改,鋪開 agent 已收增補訊息
- 二元欄表頭(2026-07-07 14:0x 使用者):值為二元(使用/未使用、是/否)的欄,表頭用「是否」句式——「本場是否使用」「是否採納」「是否已提交」;樣板+前言已改,鋪開 agent 已收增補,規範檔須寫入此規則
- 表格列標籤(2026-07-07 14:1x 使用者):表內「階段 N」一律寫粗體「**N**」(表頭已有「階段」不重複);子階段 1.1 照舊;內文散文保留「階段 N」;樣板+前言已改,鋪開 agent 已收第三條增補
- 「口徑」入禁用行話(2026-07-07 14:2x,使用者問「口徑是什麼」→白話化):→「計分方式」「取整計分」等寫法;樣板 5 處+前言 1 處已改,鋪開 agent 已收第四條增補
- 術語表收錄標準(2026-07-07 15:1x 使用者):解釋區只收「真的有必要解釋」的詞=業界/教科書既有術語或不可替代記號;**嚴禁自創標籤**(可一句白話講清就直接白話寫)。已砍自創詞:探針、第0折代理評估、OOF-only、深耕時期(正文白話化);留真術語:巢狀驗證、rank-average(入前言)。收斂 agent 執行 10 場中
- 用詞(2026-07-07 15:2x 使用者):「術語」一律寫「專業術語」(標題/表頭/內文);前言9處+樣板4處+規範三檔26處已改,收斂 agent 已收增補並代commit規範檔
- 樣板已開給使用者最終驗收,未反對即視為核可(使用者已授權自主推進)
**執行中 agents**(卡住逾40分可用 SendMessage 喚醒;曾有agent把中途訊息當注入而拒絕→重要變更一律重派全新任務書,樣板檔優先於文字指示):
- a102fcfad7207639d:統一格式鋪開(規範+9場S3+s4e1+SUMMARY 共12文件+commit;已收7輪修正訊息,樣板為準)
- a95faaae9ed070e96:跨季 s4e11 單元(四層+報告+LB;報告以樣板為準)
**佇列**:s5e10→s6e1→s6e2(各:四層+報告+提交,參照 run_s4e1_v3.py driver 與 s4e1 單元寫法)→ benchmark/SUMMARY 擴15場(階段語言)→ Phase J(idea_bank→v4注入鉤→階段5=樹搜尋+外部想法注入,2-3場歸因+掃描)
**其他**:s4e1 已完成(2e504bf,四層 0.8922→0.894302→0.894354→0.894393,LB關閉CV-only);併行session協定見上;提交已授權(token在~/.kaggle/kaggle_api_token.txt);今晚使用者下班VS Code視窗必須保持開啟
