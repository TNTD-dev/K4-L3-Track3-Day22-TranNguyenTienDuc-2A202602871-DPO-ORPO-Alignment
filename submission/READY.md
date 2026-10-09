# Kiểm tra bài nộp — 09/10/2026

- NB0–NB4 runtime complete; giữ output, bốn ảnh, split fingerprint, config và số liệu.
- REFLECTION §3/§5/§6/§7 lần lượt 254/224/275/243 từ theo đếm whitespace.
- `make verify PY=python3` mã 0; log: `verify-final.log`.
- 67 test CPU đạt; log: `tests-final.log`.
- Bonus có bằng chứng: NB3b +8, GRPO +8, β-sweep +6, chấm chéo +4, HF Hub +3 = 29 danh nghĩa; rubric giới hạn +20.
- GGUF/benchmark chưa hoàn thành vì lỗi môi trường, được ghi rõ, không tick thành công.
- Không commit weights, tokenizer JSON lớn hay Secrets. DPO config và dữ liệu runtime giữ nguyên byte; xác minh hash qua provenance.
- GitHub repo public là link nộp LMS. Điểm cuối do người chấm quyết định.

Phân tích: `REFLECTION.md`; kết quả tái tính: `analysis.json`; nguồn: `provenance/import.json`.
