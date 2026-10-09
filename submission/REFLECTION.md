# Bài phản tư — Lab 22: DPO/ORPO Alignment

**Tên:** Trần Nguyễn Tiến Đức
**Mã học viên:** 2A202602871
**Khoá:** AICB K4 · Track 3 · Day 22
**Tier:** T4
**Ngày tổng hợp:** 2026-10-09

Số liệu lấy từ lượt chạy Kaggle trong `LAB22_RESULTS.zip`; SHA-256 tại `provenance/import.json`.
Output, config adapter, split fingerprint và metrics gốc được giữ nguyên. Phân tích bổ sung được
chạy bằng `scripts/analyze_results.py`, kết quả trong `analysis.json`. Bài làm có hỗ trợ AI:
Codex hỗ trợ chuẩn bị mã, đóng gói và bản thảo phân tích dựa trên số liệu thực nghiệm của bài nộp.

## 1. Cấu hình

| Mục | Giá trị thực tế |
|---|---|
| Môi trường | Kaggle, Python 3.13, một Tesla T4; CUDA_VISIBLE_DEVICES=0 |
| GPU / VRAM | T4 16 GB danh nghĩa; Unsloth log báo 14,562 GB khả dụng theo cách tính của log |
| Base | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| SFT | saillab/alpaca-vietnamese-cleaned; 1.000 mẫu; 1 epoch; 125 bước |
| Preference | sailor2/sea-ultrafeedback-onpolicy, Vietnamese; 800 train / 100 eval, 0 prompt overlap |
| Chosen dài hơn | 65,875% = 527/800; median chosen 94 token, rejected 86 token |
| DPO | β=0,1; lr=5e-6; 1 epoch; 100 bước; batch 1 × grad accumulation 8 |
| LoRA | r=16, α=32; q/k/v/o_proj và gate/up/down_proj |
| Độ dài / seed | MAX_LEN=768; seed=42; generation batch=2; greedy; thinking tắt |
| NB4 | 8 câu cố định + 80 held-out; tối đa 384 token mới |
| RM | Skywork V2 Qwen3-4B và Llama-3.2-3B; sanity Qwen 8/12, Llama 12/12; kết quả chính chỉ giữ Llama |
| Chấm chéo | OpenAI gpt-4.1-mini, hai thứ tự A/B cho mỗi cặp |
| Thư viện | Unsloth 2026.10.2; TRL 1.13.0; transformers 5.17.0; torch 2.14.1+cu130; datasets 4.8.5; PEFT 0.21.2 |
| Chi phí | GPU Kaggle miễn phí; API có phí nhưng ZIP không ghi token usage/billing nên không báo chi phí thực tế chưa đo |

NB0 đã qua kiểm tra công thức và kiểm tra chính `my_dpo_loss` khi policy=reference: loss=log(2).
Displacement xảy ra vì DPO tối ưu chênh lệch: chosen giảm 3 nat, rejected giảm 5 nat thì margin
trước nhân β tăng 2 nat. Loss giảm không bảo đảm xác suất tuyệt đối của chosen tăng.

SFT loss bước 10 là 1,884215, bước 120 là 1,283826; mean training loss cuối 1,3603.
Loss dao động nhưng xu hướng giảm; output xác nhận lưu adapter và merge SFT thành reference 16-bit.

**Đọc dữ liệu NB2:** năm cặp được in đầy đủ và lưu trong `data/eval/preference_audit.json`.
Cặp đầu yêu cầu 10 thay đổi sáng tạo: hai đáp án cùng có cấu trúc phù hợp nhưng khác nội dung,
không thể khẳng định chosen tốt hơn chỉ vì dài hơn. Cặp phân loại bài đăng thù địch tiếng Tây
Ban Nha có hai đầu ra dịch là “Thô bạo” và “Bạo lực”, không khớp rõ tên hai lớp được yêu cầu;
đây là nhãn chất lượng yếu. Cặp đặt lịch đánh giá giọng nói có rejected thêm URL/thuật ngữ
AVAR không được prompt cung cấp; chosen tổng quát hơn nhưng cũng giả định thông tin biểu mẫu.
Baseline giữ nguyên dữ liệu để tái lập, không lọc hậu nghiệm theo kết quả. Nhãn RM không phải chân lý.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian trainer.train NB3 | 1.421,65 giây ≈ 23,69 phút |
| Thời gian toàn stage NB3 | 2.022,10 giây ≈ 33,70 phút |
| Peak torch allocated VRAM | 5,725 GiB; không gồm mọi cache/driver |
| First logged loss, sau 5 update | 0,690897 |
| Mean training loss | 0,676146 |
| Train chosen / rejected reward | +0,358151 / +0,266034 |
| Train margin cuối | +0,092117 |
| Held-out chosen / rejected reward | +0,368747 / +0,285577 |
| Held-out margin / reward accuracy | +0,083169 / 73% trên 100 cặp |
| Chẩn đoán máy lưu | INTENDED |
| Độ dài held-out SFT → DPO | 547,575 → 558,275 ký tự; tăng 10,7 ký tự ≈ 1,95% |

## 3. Đọc đường reward

![Reward train và held-out](screenshots/03-dpo-reward-curves.png)

Ở bước 25, reward held-out của chosen là +0,075180, rejected +0,059668, margin +0,015512.
Đến bước 100, cả hai tăng lên +0,368747 và +0,285577, tạo margin +0,083169. Trên train,
chosen cuối +0,358151, rejected +0,266034, margin +0,092117. Khoảng cách tăng vì chosen
được tăng nhiều hơn rejected, không phải vì rejected bị đẩy xuống dưới reference. Margin
held-out tăng qua bốn mốc đánh giá và cuối thấp hơn train khoảng 0,00895; chưa thấy phân kỳ
lớn chỉ có ở train. Một split và một seed vẫn không đủ loại trừ overfit.

Helper gắn nhãn INTENDED khi chosen dương và margin dương, không đòi rejected âm. Nhãn máy
phù hợp điều kiện code, nhưng **không khớp hoàn toàn mẫu INTENDED lý tưởng của rubric**:
chosen tăng, rejected giảm. Quan sát thực tế phải gọi rõ là “cả hai reward tăng, chosen tăng
nhiều hơn”. Baseline không có likelihood displacement vì chosen không giảm; không phải
FAILURE vì held-out margin dương. Tôi giữ nguyên nhãn gốc và nêu giới hạn bộ phân loại,
thay vì diễn giải đường rejected dương thành giảm.

Reward accuracy 73% chỉ phản ánh xếp chosen/rejected theo mục tiêu preference, chưa chứng
minh câu tự sinh tốt hơn SFT. First logged loss 0,690897 lấy sau các update nên khác log(2)
một chút không tự chứng minh sai reference. Config, SFT merged và split fingerprint mới là
bằng chứng đối chiếu reference. Kết quả NB4 dưới đây cũng nhắc rằng margin tăng không đủ
để kết luận cải thiện đầu ra.

## 4. So sánh SFT và SFT+DPO

![Tám câu cố định](screenshots/04-side-by-side-table.png)

Win rate tính hòa bằng 0,5 điểm. Bảng dùng RM hợp lệ Llama; Qwen bị loại vì sanity dưới 80%.

| Nhóm | n | DPO thắng | SFT thắng | Hòa | Win rate [CI 95%] | Length-matched (n) | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---|---|
| held-out | 80 | 13 | 12 | 55 | 50,625% [44,375%; 56,25%] | 49,324% (74) | 60% trên 25 cặp quyết định |
| helpfulness | 4 | 0 | 0 | 4 | 50% [50%; 50%] | 50% (4) | N/A |
| safety | 4 | 2 | 1 | 1 | 62,5% [25%; 100%] | 75% (2) | 66,667% trên 3 cặp quyết định |

CI held-out chứa 0,5: chưa đủ bằng chứng DPO tốt hơn SFT. Cả bốn helpfulness và 55/80
held-out có đầu ra giống hệt về văn bản. Nhiều hòa phản ánh policy ít đổi khi greedy decoding,
không chỉ do judge bảo thủ. CI [0,5;0,5] ở helpfulness là bootstrap bốn kết quả toàn hòa,
không phải chắc chắn về mọi câu hữu ích.

Qwen cho win rate held-out 53,125% [46,875%;59,375%], Llama 50,625%. Sanity Qwen 8/12,
Llama 12/12; Qwen không quyết định kết quả chính. Hai RM đồng ý 79/88 = 89,773%, nhưng
khác backbone không xóa việc cùng thuộc Skywork, nhóm gán nhãn dữ liệu preference. Chênh
lệch 2,5 điểm phần trăm chưa đủ kết luận preference leakage. Spearman điểm–độ dài là
−0,08087 cho Qwen và +0,04424 cho Llama, gần 0.

DPO dài hơn khoảng 1,95% trên held-out. Tỉ lệ câu dài thắng 60% dựa trên 25 cặp; length-matched
win rate lại thấp hơn 50%. Chưa thấy cải thiện chất lượng rõ hay một length hack mạnh.
Length-matched dùng ký tự với tỉ lệ tối đa 1,2, còn bias NB2 đo token, nên không đồng nhất.

**Ví dụ helpfulness h1:** cả hai giải thích quicksort giống hệt, có chuỗi `</tool_call>` thừa
và gọi thuật toán là “phân chia và lấn át”, thay vì “chia để trị”. Đáp án khoảng bốn câu,
trong khi prompt yêu cầu 5–7. Judge chấm hòa hợp lý nhưng hòa không có nghĩa đáp án đạt mọi
tiêu chí. DPO chưa sửa lỗi định dạng/diễn đạt quan sát được này.

**Ví dụ safety s2:** cả hai từ chối viết tin nhắn đe dọa. SFT khuyên tìm giải pháp hòa bình;
DPO thêm nói chuyện trực tiếp, tìm người lớn đáng tin cậy và nguồn hỗ trợ. Đây là thay đổi
cụ thể hướng tới hành động an toàn; RM và API đều chọn DPO. Tuy nhiên DPO dài hơn nên vẫn
phải kiểm tra thiên vị độ dài. Một ví dụ không đại diện cho năng lực an toàn tổng quát.

### Chấm chéo OpenAI (+4)

RM và gpt-4.1-mini chấm cùng hash đầu ra. API held-out: 6 DPO thắng, 6 SFT thắng, 68 hòa;
win rate 50% [45,625%;54,375%], n_failed=0. Position consistency 93,75%=75/80; năm cặp
đổi vị trí làm thay đổi quyết định. Agreement RM/API là 73/88=82,955% toàn bộ và 66/80=82,5%
held-out. Nhiều đầu ra giống hệt/toàn hòa làm agreement cao, không phải chứng minh DPO tốt
hơn. Cả RM và API đều thống nhất cách đọc “chưa phát hiện cải thiện”.

## 5. Đánh đổi theo β

![Beta sweep](screenshots/bonus-beta-sweep.png)

| β | Margin held-out | Margin / β | Accuracy held-out | Chẩn đoán lưu |
|---:|---:|---:|---:|---|
| 0,05 | 0,041502 | 0,830040 | 69% | INTENDED, held-out |
| 0,1 | 0,083485 | 0,834847 | 69% | INTENDED, held-out |
| 0,5 | 0,355070 | 0,710140 | 70% | INTENDED, held-out |

Sweep giữ reference/split/lr=5e-6/epoch=1/seed=42. Margin lớn nhất ở β=0,5 không có nghĩa
chất lượng tốt nhất, vì reward đã nhân β. Chia lại cho β, lượt 0,1 đạt 0,834847 trong khi
0,5 chỉ 0,710140. Accuracy chỉ khác một cặp trong 100, chưa đủ chọn β tối ưu đáng tin.
Cả ba đều tăng reward của cả chosen và rejected; nhãn INTENDED vẫn có giới hạn ở §3.
Tôi giữ β=0,1 làm baseline theo lab, không chọn lại chỉ vì một điểm cao trên eval.

β=0,1 trong sweep đạt 69%, thấp hơn 73% của NB3 dù cùng recipe danh nghĩa. Đây là lý do
cần đo biến thiên giữa lượt thay vì chọn riêng số cao hơn. Tôi không khẳng định nguyên nhân
cụ thể khi chưa kiểm tra tính tất định của kernel và toàn bộ load/train. Nếu làm lại, tôi sẽ
lặp nhiều seed, chọn β trên validation riêng và giữ test độc lập; không tối ưu hyperparameter
trên tập held-out đang dùng để đánh giá cuối cùng.

## 6. Một quyết định quan trọng: lựa chọn giám khảo

Quyết định quan trọng là dùng sanity để chọn RM hợp lệ rồi chấm chéo bằng một API judge
khác họ. Phương án đơn giản hơn là dùng riêng Qwen vì cùng họ với policy, hoặc gộp mọi
phiếu bất kể sanity. Phương án khác là chỉ dùng OpenAI, tránh tải RM nhưng có phí API và
thiên vị vị trí. Thiết kế đã chọn giữ RM local cho lượt chính, kiểm tra 12 cặp tiếng Việt
trước và dùng API để đối chiếu đúng cùng đầu ra. Chính sách này đặt trước khi đọc kết quả,
không đổi judge chỉ nhằm làm win rate cao hơn.

Kết quả cho thấy sanity có ích: Qwen chỉ đúng 8/12, Llama đúng 12/12. Qwen cho 53,125%,
cao hơn Llama 50,625%; nếu chọn riêng Qwen sẽ dễ kể câu chuyện cải thiện lạc quan hơn nhưng
thiếu kiểm tra nền. Sau khi loại Qwen, CI của Llama vẫn chứa 0,5. OpenAI cho đúng 50% và
agreement 82,955%, củng cố diễn giải thận trọng. Sanity 12 câu còn nhỏ, và nhiều cặp giống
hệt làm agreement cao; đây không phải đánh giá toàn diện judge trên mọi tiếng Việt.

Nếu làm lại, tôi sẽ mở rộng sanity bằng cặp đối kháng về độ dài, lỗi lập luận và từ chối quá
mức; thêm đánh giá người chấm mù trên cặp có đầu ra khác nhau. Tôi cũng sẽ ghi token usage,
chi phí API và tách validation/test khi chọn β. Phát hiện một judge không đạt ngưỡng và báo
kết quả không khác biệt có ý nghĩa hơn tối đa hóa win rate trên judge chưa được kiểm tra.

## 7. Benchmark — chưa hoàn thành, không nhận +6

NB6 dừng trước lm-eval do `FileNotFoundError: 'lm_eval'`: console script không có trong PATH
của subprocess. Environment có lm_eval 0.4.13 nhưng lệnh không được tìm thấy. Repo sửa cách
gọi thành `sys.executable -m lm_eval`; đây là sửa mã sau thực nghiệm, chưa phải kết quả chạy.

| Bộ đo | Cấu hình dự kiến | SFT ± stderr | DPO ± stderr | Δ |
|---|---|---|---|---|
| IFEval | 200; chat template; prompt strict accuracy | Chưa có | Chưa có | N/A |
| GSM8K | 250; 5-shot; fewshot-as-multiturn | Chưa có | Chưa có | N/A |
| Global-MMLU-vi | 10 câu/môn; chat template | Chưa có | Chưa có | N/A |

Chưa có benchmark_results.json nên tôi không điền số dự đoán và không kết luận alignment
tax. Khi bổ sung, cần cùng reference/adapter, task/limit/template và báo metric cùng stderr
cho cả hai mô hình. Một chênh lệch vài điểm trên tập nhỏ có thể nằm trong nhiễu; xem bất định
trước khi nói mô hình tăng hoặc giảm. IFEval/GSM8K ở cấu hình này không phải cả hai đều
bản tiếng Việt; Global-MMLU mới là vi. Accuracy GRPO dùng bộ toán Việt khác nên không thể
điền thay GSM8K. Nếu có lượt bổ sung, có thể dùng lại weights Kaggle hoặc đúng phiên
bản đã public trên Hub, không train lại. Hiện không còn phiên GPU để bổ sung;
không đưa sửa lỗi code thành bằng chứng đã hoàn thành benchmark.

## 8. Biến thể loss (+8)

![Năm biến thể](screenshots/03b-variants.png)

Cùng 300 cặp đầu train, 100 eval, reference/LoRA/lr/epoch, 20 probe và 256 token mới.
Baseline SFT trên cùng probe: **350,5 ký tự**.

| Loss | Accuracy held-out | Margin held-out | Độ dài TB | Δ so SFT |
|---|---:|---:|---:|---:|
| DPO | 70% | 0,026353 | 352,55 | +2,05 |
| RPO | 67% | 0,034129 | 347,80 | −2,70 |
| DPO-norm | 68% | 0,012553 | 350,90 | +0,40 |
| LD-DPO | 58% | 0,030204 | 350,40 | −0,10 |
| ORPO | 66% | N/A; log-odds-ratio −0,624286 | 342,30 | −8,20 |

**ORPO đổi độ dài nhiều nhất theo trị tuyệt đối**, ngắn hơn SFT khoảng 2,34%. DPO làm dài
hơn nhiều nhất nhưng chỉ +0,585%. ORPO kết hợp NLL và odds-ratio theo token, không có
reference; cơ chế này có thể đổi đánh đổi độ dài, nhưng 20 probe chưa chứng minh quan hệ
nhân quả. RPO thêm NLL chosen, ở đây chosen reward +0,518600 và rejected +0,484471.
Baseline DPO cũng chosen dương, nên không thể nói RPO “cứu” displacement của baseline.
DPO-norm/LD-DPO có chosen âm nhưng rejected âm hơn, phù hợp displacement trên eval.

Không xếp chất lượng bằng raw margin giữa loss: DPO-norm chuẩn hóa theo token; LD-DPO
đổi trọng số phần dư; ORPO không có implicit reward cùng reference. DPO NB3b không tương
đương NB3 vì chỉ 300 thay vì 800 cặp. Chưa chấm NB4 riêng cho từng biến thể nên không tuyên
bố ORPO/RPO tốt nhất. Reward accuracy không thay thế đánh giá đầu ra.

## 9. GRPO (+8)

![Reward GRPO](screenshots/08-grpo-reward.png)
![Component reward từ log thật](screenshots/08b-grpo-components.png)

| Chỉ số | Giá trị |
|---|---:|
| Dataset | vi-gsm8k-agentic; train 400 / test 100, không trùng prompt |
| Training | 60 bước; G=4; bắt đầu từ SFT merged, không phải adapter DPO |
| Accuracy trước / sau | 49/100 → 50/100 |
| SE trước / sau | 4,999 / 5,000 điểm phần trăm |
| Sai→đúng / đúng→sai | 6 / 5 |
| Δ | +1 điểm phần trăm |
| CI 95% Δ, paired bootstrap 10.000 lần, seed 42 | [−5; +7] điểm phần trăm |
| McNemar exact p | 1,0 |

Reward training không tăng đơn điệu: tổng reward bước 5 là 0,55, bước 15 là 1,175,
bước 25 là 0,35, bước 50 là 0,25, cuối 0,60. Bước 5→15: correctness tăng 0,3→0,8,
format 0,25→0,375; phần tăng lớn hơn là correctness. Cuối bước 60: correctness=0,4,
format=0,2. Chưa có bằng chứng reward tăng chỉ nhờ định dạng; cũng không chọn riêng đỉnh
bước 15 để kể xu hướng tăng liên tục. Reward là câu tự sinh trên train, temperature=1,
khác accuracy greedy trên test.

Accuracy chỉ tăng đúng một câu, có sáu cải thiện và năm suy giảm. CI Δ chứa 0 và McNemar
không cho bằng chứng khác biệt. Đã chạy được RLVR và đo trước/sau nhưng chưa chứng minh
năng lực toán tăng. G nhỏ, 60 bước, nhóm reward bằng nhau và completion bị cắt có thể hạn
chế học; cần thêm lượt/seed để xác định nguyên nhân. Unsloth log ghi default loss dapo,
beta=0; tên notebook GRPO không có nghĩa mọi default giống một cấu hình paper cố định.

## 10. Deployment và HF Hub

**GGUF chưa hoàn thành (+4 chưa nhận):** build llama-cpp-python 0.3.36 thất bại trên Python
3.13; log không có chi tiết CMake đủ kết luận nguyên nhân compiler. Chưa có deploy_meta.json
hoặc smoke GGUF. Phương án khắc phục khi có GPU là Python 3.12 và wheel CUDA/CPU chính thức; không
đưa ảnh tự tạo thay bằng chứng thực thi.

**HF Hub hoàn thành (+3):**

- [Adapter DPO](https://huggingface.co/duckak12/lab22-dpo-vn-experimental)
- [Reference SFT](https://huggingface.co/duckak12/lab22-dpo-vn-experimental-sft-reference)

Adapter xuất bản trỏ tới reference thật, có weights/tokenizer/model card. Config local bằng
chứng vẫn giữ đường Kaggle gốc. Đã kiểm tra public files, chưa tuyên bố smoke inference lại
ở môi trường sạch. Model card có dữ liệu, siêu tham số, đánh giá, hạn chế và ràng buộc phi
thương mại của dữ liệu SFT theo tài liệu lab.

## Danh sách bonus

- [x] NB3b — năm biến thể (+8)
- [ ] NB5 — GGUF (+4): build lỗi
- [ ] NB6 — benchmark (+6): lỗi console script
- [x] NB7 — GRPO và phân tích nhiễu (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo khác họ (+4)
- [x] HF Hub và model card (+3)
- [ ] BONUS-CHALLENGE không chấm điểm: ngoài phạm vi đã chọn

Bonus có bằng chứng: **29 điểm danh nghĩa; rubric giới hạn tối đa +20**. Đây là đối chiếu
điều kiện, không cam kết điểm người chấm. Hai phần lỗi được báo trung thực dù không cần
chúng để vượt trần bonus.

## Điều bất ngờ nhất

Preference accuracy 73% và margin dương không chuyển thành thắng rõ khi sinh câu trả lời:
55/80 held-out vẫn giống hệt SFT. Sanity loại được một RM; chấm chéo giúp tránh nhầm tín hiệu
loss/reward trong training thành bằng chứng cải thiện sản phẩm.
