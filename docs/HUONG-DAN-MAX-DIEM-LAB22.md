# Cách làm Lab 22 để hướng tới điểm tối đa

Đối chiếu tài liệu và mã nguồn trong repo ngày 08/10/2026. Đây là hướng dẫn thực hiện; chưa phải kết quả thực nghiệm của bài nộp.

## 1. Mục tiêu điểm và thứ tự ưu tiên

**Theo rubric: 100 điểm bắt buộc + tối đa 20 điểm bonus = 120 điểm thô.** Tài liệu không nói LMS có quy đổi hoặc giới hạn về 100 hay không. Lab chiếm 30% Daily Lab của Track 3. Chấm theo bằng chứng và cách giải thích, không đòi mô hình phải đạt một win rate cụ thể.

Các bonus có điểm cộng lại là 39, nhưng chỉ tính tối đa 20. Có hai mục tiêu khác nhau:

| Mục tiêu | Cách thực hiện | Điểm bonus danh nghĩa |
|---|---|---:|
| Lấy đủ trần bonus bằng notebook có sẵn | NB3b (+8) + NB6 (+6) + NB7 (+8); không cần API key | 22 → tối đa 20 |
| Một phương án đạt trần +20, tránh GRPO | NB3b (+8) + NB6 (+6) + NB5 (+4) + chấm chéo (+4) | 22 → tối đa 20 |
| Đạt trần +20 bằng 4 phần, không cần API key | NB3b (+8) + β-sweep (+6) + NB5 (+4) + HF Hub (+3) | 21 → tối đa 20 |
| Hoàn thành toàn bộ phần có chấm điểm | NB3b, NB5, NB6, NB7, β-sweep, chấm chéo, HF Hub | 39 → tối đa 20 |

**Đề xuất theo yêu cầu “cả bonus luôn”:** hoàn thành NB0–NB4, lưu bằng chứng và viết phản tư trước; sau đó làm toàn bộ bảy phần bonus. Nếu thời gian GPU thiếu, chọn một tổ hợp vượt 20 ở bảng trên. Những tổ hợp này là gợi ý tổ chức công việc dựa trên rubric, chưa được đo chi phí thực tế.

**Riêng `BONUS-CHALLENGE.md`: không chấm điểm, không hạn nộp.** README dẫn tới file này để xem bonus, nhưng nội dung thực tế là dự án ứng dụng tự chọn. Các bonus có điểm phải tra `rubric.md`, notebook tương ứng và REFLECTION. Đừng dùng dự án này thay cho NB0–NB4 hoặc các bonus có điểm.

Nguồn: `rubric.md`, `README.md` §2–§5, `BONUS-CHALLENGE.md`.

## 2. Chuẩn bị môi trường và thứ tự chạy

T4 và BigGPU được chấm cùng tiêu chí; dùng mô hình 8B không tự động được thêm điểm. Với cấu hình bài hiện tại, chọn Colab T4 là đường mặc định. Trên Apple Silicon, NB0 và test CPU chạy được, nhưng pipeline Unsloth/bitsandbytes của lab cần NVIDIA CUDA; không chuyển thẳng sang MPS.

| Cấu hình | T4 mặc định | BigGPU |
|---|---|---|
| Notebook | `colab/Lab22_DPO_T4.ipynb` | `colab/Lab22_DPO_BigGPU.ipynb` |
| Mô hình | Qwen3-4B-Instruct-2507, 4-bit | Qwen3-8B, 4-bit, thinking tắt |
| SFT | 1.000 mẫu tiếng Việt | 2.000 mẫu |
| Preference train / eval | 800 / 100 cặp | 3.500 / 200 cặp |
| Độ dài tối đa | 768 token | 1.024 token |
| Câu held-out NB4 | 50 mặc định | 100 mặc định |

Cấu hình DPO khởi đầu: β = 0,1; lr = 5e-6; 1 epoch; LoRA r = 16, α = 32; seed = 42. Dùng cấu hình này làm baseline rồi mới đổi từng yếu tố. OOM thì giảm `MAX_LEN` xuống 512 **trước khi tạo dữ liệu**, ghi rõ thay đổi và số cặp thực tế còn lại. Không dùng lượt chạy nhanh `PREF_TRAIN=200` làm thay thế âm thầm cho bài đầy đủ.

Thứ tự thực hiện đề xuất:

1. Setup → NB0 → NB1 → NB2 → NB3 → NB4.
2. Tải notebook có output, ảnh, metadata và dữ liệu về; viết REFLECTION §1–§4, §6.
3. NB3b → NB5 → NB6 → NB7; làm β-sweep, chấm chéo, HF Hub.
4. Hoàn thiện các mục phản tư bonus, chạy verify, kiểm tra file đã commit, nộp LMS.

Trong Colab, **NB3b nằm trước NB4 dù là bonus**. Có thể chạy luôn nếu muốn làm tất cả; nếu muốn chốt phần chính trước thì bỏ qua riêng đoạn NB3b, giữ cell giải phóng GPU, chuyển tới NB4. “Chạy tất cả” trên bản Colab hiện tại chạy cả bonus, không chỉ NB0–NB4.

Máy NVIDIA có đủ môi trường: điền TODO NB0 rồi chạy các lệnh sau tại gốc repo. Không chạy `make -j` cho pipeline dùng cùng GPU.

```bash
bash setup-laptop.sh
make smoke
make pipeline
# Sau khi lưu bằng chứng phần chính:
make variants
make deploy
make bench
make grpo
make beta-sweep
# Điền REFLECTION trước bước cuối:
make verify
```

`make pipeline-full` = phần chính + NB3b/NB5/NB6/NB7; chưa bao gồm β-sweep, chấm chéo và HF Hub. Thời gian tài liệu ước tính phần chính 1,5–2 giờ trên T4; NB6 khoảng 40–60 phút; NB7 khoảng 40 phút. **CHANGELOG nói chưa chạy end-to-end trên GPU**, nên dành thời gian dự phòng; đây không phải cam kết thời lượng.

Colab hết phiên sẽ mất `/content`. Lưu cả trọng số SFT/DPO ở nơi bền vững nếu muốn tiếp tục bonus trong phiên khác; gói bằng chứng nhỏ đủ để nộp nhưng không đủ để tiếp tục huấn luyện hoặc xuất GGUF. Restart runtime còn ổ đĩa thì có thể setup lại và tiếp tục từ bước lỗi; runtime mới đã mất file thì phải khôi phục trọng số hoặc chạy lại NB1 trở đi.

Nguồn: `lab22/config.py`, `Makefile`, `HARDWARE-GUIDE.md`, `scripts/build_colab.py`, `CHANGELOG.md`.

## 3. Lấy đủ 100 điểm phần chính

| Phần | Điểm | Việc làm và điều kiện cần chứng minh |
|---|---:|---|
| NB0 — loss | 6 | Điền `my_dpo_loss`, khớp công thức tham chiếu; kiểm tra chính hàm tự viết cho policy = reference ra log(2) |
| NB0 — giải thích | 4 | Giải thích likelihood displacement bằng ví dụ số, không chỉ chép tên hiện tượng |
| NB1 — SFT | 8 | Loss giảm; lưu adapter SFT và `models/sft-merged/`; giữ output merge thành công |
| NB2 — chia dữ liệu | 8 | Train/eval không trùng prompt; assert qua; đọc và nhận xét ít nhất 3 cặp chosen/rejected |
| NB2 — thiên vị độ dài | 4 | Ảnh `02b-pref-length.png` và tỉ lệ chosen dài hơn, đo theo token |
| NB3 — đúng reference | 6 | Huấn luyện LoRA mới trên SFT merged; adapter config trỏ đúng mô hình này, giữ fingerprint split |
| NB3 — biểu đồ | 10 | Tách chosen/rejected trên cả train và held-out; kèm margin; không chỉ vẽ loss hoặc margin train |
| NB3 — chẩn đoán | 8 | REFLECTION §3 giải thích INTENDED / DISPLACEMENT / FAILURE / AMBIGUOUS bằng số liệu thực |
| NB4 — sinh đầu ra | 6 | 8 câu cố định và ít nhất 50 prompt held-out khác nhau; SFT và DPO cùng cấu hình sinh |
| NB4 — judge | 10 | Win rate, CI 95%, sanity, câu dài thắng, length-matched win rate; đọc per_judge |
| Phản tư | 20 | §3, §4, §6 có số liệu và phân tích của lượt chạy thực; nêu 2 ví dụ hữu ích/an toàn |
| Tái lập | 5 | Notebook đã chạy còn output; môi trường sạch chạy được theo chỉ dẫn đã ghi |
| Verify | 5 | Chạy `make verify`, mã thoát 0; giữ log làm minh chứng |
| Tổng | 100 | Mỗi dòng cần có bằng chứng có thể đọc từ bài nộp |

### NB0: phần cần tự triển khai

Trong `notebooks/00_dpo_loss_from_scratch.py`, thay `return None` bằng công thức này:

```python
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    margin = beta * ((pc - rc) - (pr - rr))
    return -torch.nn.functional.logsigmoid(margin).mean()
```

Giữ assert so với tham chiếu và thêm kiểm tra chính hàm này với policy = reference. Cell hiện tại dùng hàm thư viện để minh họa log(2), không kiểm tra hàm TODO của bạn trong tình huống đó.

```python
assert torch.allclose(
    my_dpo_loss(same, same - 3, same, same - 3),
    torch.tensor(math.log(2), dtype=same.dtype),
    atol=1e-6,
)
```

Ví dụ trả lời câu hỏi: chosen từ −20 thành −23, rejected từ −22 thành −27. Chosen giảm 3, rejected giảm 5, nên chênh lệch so reference tăng 2 nat; với β = 0,1, reward chosen = −0,3, reward rejected = −0,5, margin = +0,2. DPO chỉ tối ưu chênh lệch này nên loss có thể giảm dù chosen giảm. Đây là ví dụ minh họa, không dùng làm số liệu huấn luyện của bạn.

### NB2: phải bổ sung bằng chứng đọc dữ liệu

Mã hiện tại chỉ `print(train_ds[0])`, chưa đủ yêu cầu 3 cặp. In đầy đủ ít nhất 3 cặp, tốt hơn là 5 cặp cố định theo seed; ghi nhận xét chosen có đúng hơn, hữu ích hơn, an toàn hơn hay chỉ dài hơn. Giữ output và ghi 3 nhận xét ngắn trong REFLECTION. Nếu lọc thêm, làm trước NB3; báo tiêu chí lọc và kích thước cuối.

### NB3: cách phân tích đủ ý

Lấy `end_chosen_reward`, `end_rejected_reward`, `end_reward_gap` cho train và các trường `eval_*` cho held-out trong `dpo_metrics.json`. So sánh dấu và xu hướng hai phía. Reward accuracy này đo khả năng xếp chosen cao hơn rejected; **khác** win rate đầu ra DPO so với SFT ở NB4.

Chosen tăng/rejected giảm/margin tăng là INTENDED. Chosen âm nhưng rejected giảm mạnh hơn là displacement. Margin held-out ≤ 0 là FAILURE theo rubric. Train cải thiện nhưng held-out không cải thiện là dấu hiệu overfit. Nếu ít điểm log hoặc đường phẳng, nói rõ giới hạn thay vì ép kết quả vào nhóm tốt.

Reward lý thuyết bắt đầu 0, loss khởi tạo log(2), nhưng log đầu sau vài update không nhất thiết bằng đúng 0,693. Không kết luận reference sai chỉ vì số đầu được ghi khác 0,693; kiểm tra config và cách khởi tạo.

### NB4: báo đúng số và đúng kết luận

- Báo riêng held-out, helpfulness (4), safety (4); tránh kết luận mạnh từ nhóm chỉ 4 câu.
- Win rate trong code = (DPO thắng + 0,5 × hòa) / số cặp chấm hợp lệ; không phải chỉ DPO thắng / tổng.
- API trả sai định dạng bị đếm `n_failed`, loại khỏi mẫu; phải còn ít nhất 50 held-out được chấm hợp lệ. Có thể tăng số prompt trong phạm vi eval nếu thiếu.
- CI chứa 0,5: chưa đủ bằng chứng khác biệt. CI hoàn toàn trên 0,5: bằng chứng nghiêng về DPO trong phép đánh giá này. Kết quả kém hoặc không khác biệt vẫn có thể được chấm đầy đủ nếu thực hiện đúng và giải thích trung thực.
- RM sanity < 0,8: nêu hạn chế và kiểm tra bằng judge khác nếu có. Hai RM dùng backbone Qwen/Llama khác nhau nhưng cùng nhóm Skywork; không coi là hai đánh giá độc lập hoàn toàn.
- Đọc `per_judge`, `judge_agreement`, `longer_answer_won_frac`, `mean_chars_*`, `length_matched_win_rate` và `length_matched_n`. Length-matched ở code là tỉ lệ độ dài ký tự không quá 1,2. Nếu n = 0, báo N/A và thiếu bằng chứng, không điền 0%.
- `score_length_spearman` có thể nằm trong từng `per_judge` thay vì summary hội đồng; tìm đúng vị trí. Với API, báo `position_consistency` khi đổi A/B.
- Chọn 1 ví dụ helpfulness và 1 safety; trích đầu ra thật từ `side_by_side.jsonl`, giải thích thay đổi cụ thể và khả năng judge đánh sai.

Nguồn: `rubric.md`, NB0–NB4, `lab22/judge.py`, `lab22/modeling.py`.

## 4. Toàn bộ bảy bonus có chấm điểm

### A. NB3b — 5 biến thể loss (+8)

Chạy đủ DPO, RPO, DPO-norm, LD-DPO, ORPO. Giữ cùng SFT merged, cùng tập dữ liệu, seed và ngân sách huấn luyện; NB3b T4 dùng 300 cặp, BigGPU dùng 1.000. So các biến thể trong NB3b với baseline DPO cùng ngân sách của NB3b, tránh so trực tiếp với NB3 đã train 800/3.500 cặp.

Nộp notebook có output, `adapters/variants/variants_summary.json`, `03b-variants.png`, REFLECTION §8. Tính margin cho các dòng DPO từ chosen − rejected nếu cần; ORPO không có cùng thang implicit reward thì ghi N/A hoặc log-odds đúng tên. Không xếp hạng chất lượng chỉ bằng raw margin giữa các loss.

Giải thích biến thể nào đổi độ dài nhiều nhất dựa trên số thực. RPO thêm NLL chosen; DPO-norm chuẩn hóa log-prob theo token; LD-DPO giảm trọng số phần token vượt độ dài chung; ORPO kết hợp SFT và odds ratio. Để biết “thay đổi” so SFT, sinh baseline SFT trên cùng 20 probe prompt và cùng giới hạn 256 token; NB3b hiện chỉ ghi độ dài các biến thể.

Nếu chấm thêm adapter biến thể bằng NB4: NB3b chưa lưu `split.json`, nên NB4 sẽ từ chối. Cần bổ sung lưu fingerprint lúc train bằng helper hiện có và giữ nguyên split; không tạo fingerprint cho một adapter có lịch sử train không xác định. `VARIANT_TRAIN` được nhắc trong comment nhưng code dùng `C.TIER.variant_train`, chưa đọc biến môi trường này.

### B. NB5 — GGUF SFT+DPO (+4)

Nạp `adapters/dpo` ở 16-bit, kiểm tra có tensor `lora_`, xuất Q4_K_M, chạy llama-cpp-python. So HF/GGUF cùng prompt đã áp chat template, thinking tắt, cùng giới hạn token. So nội dung, phong cách và sai khác do lượng tử hóa; không bắt buộc câu chữ giống hệt.

Nộp `data/eval/deploy_meta.json`, notebook output và **ảnh chụp thủ công** `06-gguf-smoke.png` có tên Q4_K_M + câu trả lời. File GGUF lớn bị gitignore; giữ ngoài Git, cung cấp đường tái tạo hoặc link tải nếu muốn người khác chạy demo. Không lấy adapter SFT để xuất thay DPO. Comment “+3” cho thử nhiều lượng tử hóa trong NB5 không phải mục điểm độc lập trong rubric hiện tại.

### C. NB6 — benchmark (+6)

Chạy cả SFT và SFT+DPO trên IFEval, GSM8K, Global-MMLU-vi, có chat template; GSM8K dùng fewshot-as-multiturn. T4 mặc định: IFEval 200, GSM8K 250, Global-MMLU-vi 10 **mỗi môn**. IFEval và GSM8K này không phải cả hai đều là tiếng Việt; phần MMLU mới là vi.

Nộp `benchmark_results.json`, `07-benchmark-comparison.png`, notebook output, REFLECTION §7 ≥ 150 từ. Báo metric, sample limit, SFT ± stderr, DPO ± stderr, Δ. Đọc chênh lệch so nhiễu như tài liệu yêu cầu. Có thể bổ sung SE(Δ) ≈ √(SE_SFT² + SE_DPO²) nếu giả định độc lập, nhưng cùng bộ câu hỏi là đánh giá ghép cặp nên đây chỉ là xấp xỉ; bootstrap ghép cặp cần dữ liệu theo câu. Không gọi Δ nhỏ là tiến bộ chắc chắn. Đề cập alignment tax nếu điểm giảm và phân biệt với nhiễu.

### D. NB7 — GRPO (+8)

Chạy reward kiểm chứng đáp số và reward định dạng trên vi-gsm8k-agentic. T4: train 400, test 100, 60 bước, G = 4. Giữ split không trùng. **NB7 bắt đầu từ SFT merged, không nối tiếp adapter DPO.** Kết luận đúng là SFT → SFT+GRPO.

Nộp `adapters/grpo/grpo_metrics.json`, `08-grpo-reward.png`, output trước/sau và REFLECTION §9. Báo n, accuracy trước/sau, SE ≈ √(p(1−p)/n) cho mỗi accuracy. Với n = 100, SE tối đa khoảng 5 điểm phần trăm; mức tin cậy 95% cho một accuracy có thể rộng khoảng ±10 điểm. Muốn kết luận chênh lệch trước/sau chắc chắn cần xét nhiễu của chênh lệch; lưu dự đoán từng câu để bootstrap ghép cặp nếu làm sâu hơn.

Notebook hiện vẽ mean reward, chưa lưu đầy đủ log từng thành phần. Giữ/in component reward trong output hoặc lưu log riêng để trả lời reward định dạng hay reward đúng tăng trước. Định dạng tốt hơn chưa chứng minh giải toán đúng hơn. Không so trực tiếp accuracy này với GSM8K tiếng Anh NB6 như cùng một phép đo.

### E. β-sweep (+6)

Chạy `make beta-sweep`: β = 0,05 / 0,1 / 0,5. Cùng reference, split, seed, lr, epoch. Nộp config/metrics/split cho `adapters/dpo-b0.05`, `dpo-b0.10`, `dpo-b0.50`, ảnh `bonus-beta-sweep.png`, REFLECTION §5 giải thích ≥ 100 từ.

Bảng phải có margin held-out và reward accuracy held-out. Margin đã nhân β nên β lớn tự làm đổi thang: xem thêm margin/β, accuracy và chosen/rejected; đừng chọn β chỉ vì raw margin cao. `scripts/train_dpo.py` hiện chẩn đoán bằng train history, còn NB3 ưu tiên held-out: ghi rõ cơ sở chẩn đoán và tự đọc held-out khi viết bài.

Colab bundle không kèm Makefile/scripts cho sweep; cần lấy các file từ cùng phiên bản repo vào `/content/lab22` trước khi chạy lệnh. Sweep không nằm trong `pipeline-full`.

### F. Chấm chéo khác họ (+4)

Chấm cùng `side_by_side.jsonl` bằng hội đồng RM và một judge API khác họ; notebook gợi ý Gemini. Ghi provider và model ID thực đã dùng; giữ key trong Secrets/.env. Báo `cross_judge.agreement`, số cặp, win rate từng judge, position consistency của API và giải thích bất đồng.

Nộp `judge_results_rm.json`, `judge_results_api.json`, `judge_summary.json` có `cross_judge`. Để giữ cùng hash, sinh câu trả lời một lần rồi chạy lại riêng cell judge và tổng hợp NB4. Cập nhật cấu hình judge trong kernel đã import `lab22.config`; chỉ đổi môi trường sau import không tự cập nhật C. Nếu chạy lại toàn NB4 và hash đổi, hai lượt không được tính chấm chéo. Thiếu API key thì fallback sang RM: **không đủ bằng chứng +4 chấm chéo**.

### G. HF Hub (+3)

Upload adapter DPO cùng tokenizer cần thiết và model card. Card phải ghi base, reference SFT merged, dữ liệu, license/hạn chế, β/lr/epoch/LoRA/seed, kết quả NB4 và cách load. Thêm link Hub trong README và REFLECTION.

Adapter hiện tham chiếu đường local `models/sft-merged`: **upload mỗi LoRA không đủ để người khác load trực tiếp từ Hub**. Cần cung cấp SFT merged ở repo Hub riêng và trỏ bản adapter xuất bản tới đó, hoặc cung cấp quy trình tái tạo reference có thể làm theo. Giữ config của bản adapter bài nộp đúng reference local để verify; chuẩn bị bản phân phối riêng cho Hub. Tokenizer + đường load thực tế cần được thử trong môi trường sạch. Không khai báo model phân phối là dùng thương mại vô điều kiện khi dữ liệu SFT có CC BY-NC theo tài liệu lab.

Nguồn: `rubric.md`, NB3b/NB5/NB6/NB7, `Makefile`, `scripts/train_dpo.py`, `scripts/eval_judge.py`, REFLECTION §5/§7–§9.

## 5. Cách viết REFLECTION để tránh mất 20 điểm

Tài liệu không thống nhất độ dài: rubric ghi §3 + §6 tổng ≥ 150 từ; README/template ghi §3 ≥ 100 và §6 ≥ 150. **Làm theo mức chặt hơn: §3 ít nhất 100 từ, §6 ít nhất 150 từ**, và §7 ít nhất 150 từ nếu làm benchmark. β-sweep cần giải thích ít nhất 100 từ.

| Mục | Cách điền |
|---|---|
| Header, §1 | Tên/khóa/ngày/tier/GPU; config thực; số train/eval thực; bias độ dài; judge; chi phí thực |
| §2 | Copy số từ `dpo_metrics.json`; lấy độ dài từ đầu ra/summary NB4; thông số thời gian/VRAM từ log, đo riêng nếu chưa được ghi |
| §3 | Số train + held-out → diễn giải chosen/rejected → chẩn đoán → khả năng overfit và giới hạn |
| §4 | Bảng cả 3 nhóm; CI/sanity/bias/per_judge; 2 ví dụ đầu ra thật, chỉ rõ cải thiện hoặc suy giảm |
| §5 | Bảng đủ 3 β, phân tích tradeoff ≥ 100 từ; nếu không chạy thì 3 câu giả thuyết theo template |
| §6 | Một quyết định, phương án khác, lý do, kết quả thật, thay đổi nếu làm lại; ≥ 150 từ |
| §7–§9 | Đủ bảng và giải thích theo bonus đã làm; ghi N/A đúng nghĩa khi metric không áp dụng |
| Bonus checklist | Tick chỉ các phần đã có bằng chứng; thêm mục deployment/chấm chéo/Hub nếu template thiếu chỗ phân tích |

Một khung §6 dễ làm tốt là quyết định chọn judge: RM miễn phí nhưng cùng nhóm Skywork; API khác họ thêm chi phí và position bias; đối chiếu agreement/win rate thực rồi rút quyết định. Có thể chọn β thay thế nếu bạn đã làm sweep. Không điền số ví dụ trong template thành số đo của mình.

Nguồn: `README.md` §3, `rubric.md`, `submission/REFLECTION.md`.

## 6. Bằng chứng phải giữ và phải commit

**Phần chính:** notebook NB0–NB4 có output hoặc Colab đã chạy; 4 ảnh bắt buộc; adapter config SFT và DPO; `dpo_metrics.json`, `split.json`; preference train/eval parquet; `side_by_side.jsonl`, `judge_summary.json`, các judge results; REFLECTION hoàn chỉnh. Giữ merged SFT và trọng số adapter ở runtime/lưu trữ bền vững để verify và chạy lại, nhưng không commit trọng số lớn.

**Bonus:** notebook có output, ảnh và JSON đã nêu ở §4; link HF Hub; tài liệu phân tích. NB5 cần ảnh thủ công, không chờ notebook tự tạo ảnh đó.

Các lỗ hổng đóng gói cần xử lý trước khi nộp:

1. `.gitignore` chặn `data/pref/stats.json`; lưu tỉ lệ vào output/REFLECTION hoặc thêm ngoại lệ chính xác cho file này.
2. `.gitignore` cho phép các metadata trực tiếp dưới `adapters/<run>/`, nhưng các thư mục lồng `adapters/variants/<name>/` còn bị chặn. Kiểm tra từng file; giữ summary, cấu hình cần thiết và fingerprint nếu bổ sung; không mở chặn toàn bộ weights.
3. `make verify` cần `models/sft-merged/config.json`, nhưng toàn bộ models bị gitignore. Chạy verify trong môi trường có đầy đủ kết quả, lưu log mã 0. Một fresh clone chưa train không thể verify ngay; phải chạy pipeline hoặc khôi phục artifacts trước.
4. Adapter config có absolute path `/content/lab22/models/sft-merged` có thể bị báo WRONG REF khi mang về máy khác. Không sửa đường để che kết quả train; kiểm tra tại runtime gốc, giữ log và hướng dẫn tái lập/di chuyển reference minh bạch. Muốn verify tại checkout khác thì cần khôi phục reference và chuẩn bị config đường dẫn đúng một cách có ghi nhận, hoặc chạy lại ở đó.
5. Colab được sinh chỉ nhúng `lab22/*.py` và các cell notebook, **không nhúng Makefile, verify script, source notebooks hay REFLECTION**. Để lấy 5 điểm verify, lấy các file hỗ trợ từ đúng repo phiên bản này vào cùng gốc `/content/lab22`, điền REFLECTION ở đó, rồi chạy verify. Nếu nguồn NB0 ở repo vẫn TODO thì cập nhật cả nguồn, không chỉ cell Colab.
6. Sau khi sửa nguồn notebook/helper, `make colab` sinh lại bundle; không chạy lệnh này lên notebook duy nhất đã giữ output nếu chưa sao lưu, vì bản sinh lại có output trống. Giữ bản thực thi riêng hoặc chạy lại bundle mới.

Lệnh kiểm tra tại repo trước khi commit:

```bash
make verify
git status --short
git check-ignore -v data/pref/stats.json
git check-ignore -v adapters/variants/dpo/adapter_config.json
git ls-files notebooks colab submission data adapters
```

`git check-ignore` trả 1 khi file không bị ignore; đó không phải lỗi bài lab. `git ls-files` cho biết file đã tracked nhưng chưa đảm bảo thay đổi mới đã commit: kiểm tra status và nội dung commit cuối. **Verify mã 0 không chứng minh đủ điểm:** script chưa kiểm tra TODO NB0, notebook output, số từ phản tư, 3 cặp mẫu hay chất lượng phân tích bonus.

Chỉ file đã commit mới được chấm. Nộp URL repo public vào LMS, **không cần tạo PR**; giữ repo public đến khi có điểm. Hạn trong tài liệu là 23:59 ngày sau ngày học, muộn trừ 10%/ngày, quá 3 ngày 0 điểm. Nếu Day22 của bạn là 08/10/2026 thì mốc tương ứng là 09/10/2026 lúc 23:59; repo không xác nhận lịch học thực, đối chiếu lịch LMS.

Nguồn: `.gitignore`, `scripts/verify.py`, `scripts/build_colab.py`, `rubric.md`.

## 7. Nếu muốn làm cả BONUS-CHALLENGE không chấm điểm

Chọn 1 ứng dụng: gia sư, CSKH, trợ lý nghề, trợ lý có ranh giới an toàn, hoặc chuyển văn phong. Theo file challenge, làm thêm:

- `bonus/README.md` ≥ 400 từ: người dùng cụ thể, domain knowledge, mục tiêu, sản phẩm đầu ra.
- 200 cặp preference trong lĩnh vực, ít nhất 100 cặp tự tạo dựa trên phán đoán chuyên môn; lưu nguồn và cách chia eval.
- Lượt train có thể tái lập, adapter; GGUF nếu ứng dụng cần, demo CLI/Gradio/FastAPI chạy bằng một lệnh.
- Model card nêu dùng để làm gì, ngoài phạm vi, hạn chế và giấy phép.
- 5 tương tác mẫu; nếu chọn trợ lý lĩnh vực nhạy cảm, bộ test 20 câu (10 nên trả lời / 10 phải từ chối và chuyển tiếp), precision/recall hai nhóm theo yêu cầu challenge.
- Nhật ký dùng AI khoảng 100 từ nếu dùng nhiều; tên thành viên nếu làm nhóm; ghi đã làm challenge trong REFLECTION.

Phần này dự kiến 4–8 giờ theo tài liệu, không tăng trần bonus. Hoàn thành bài có điểm và nộp đúng hạn trước rồi làm challenge nếu muốn một sản phẩm portfolio.

## 8. Trạng thái repo tại thời điểm đọc

Đã chạy read-only `python3 scripts/verify.py`: mã thoát **1**. Thiếu SFT config, SFT merged config, preference train/eval, DPO config, side-by-side, 4 ảnh; REFLECTION header và §1/§2/§3/§4/§6 còn placeholder. NB0 vẫn `return None`. Các bonus chưa có JSON/ảnh theo kiểm tra script. Đây là repo khởi đầu chưa đủ minh chứng nộp, không phải báo cáo thực nghiệm đã hoàn thành.

Tài liệu đã đối chiếu: README, rubric, BONUS-CHALLENGE, HARDWARE-GUIDE, docs/reference, VIBE-CODING, CHANGELOG, template REFLECTION, screenshots/README và solutions/README. `solutions/` hiện chỉ có thông báo, chưa có lời giải. Kiểm tra thêm source NB0–NB7, config, judge, modeling, data, Makefile, gitignore, build_colab và verify để phân biệt yêu cầu trên giấy với hành vi code thực tế.
