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
- [x] 跨季單元 1/5 s4e1 完成(四層 0.8922→0.894302→0.894354→0.894393;tier4 樹搜尋 60 節點滿預算,node#44 37成員 mega-blend 勝 tier3;H-1 load_search_state 首次真實復原成功(前代被 kill 於播種中,5 節點逐位元載回+補種),並修復 driver「lineage 提案耗盡空轉」缺口;OOF 重現閘門 37/37 全過 max|d|=0.0 → sub_tree_best_0.89439;LB 關閉實測 403(tier3/tier4 檔各一次)→ CV-only;exp17 入帳;REPORT 7 節含 R-W9 新 §2(工具/術語/刻度)verify=0 + s4e1_REPORT.pdf)
