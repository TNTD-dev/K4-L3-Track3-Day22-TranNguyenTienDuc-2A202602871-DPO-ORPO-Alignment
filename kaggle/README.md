# Chạy Lab22 trên Kaggle và gửi kết quả về

File cần upload: **`Lab22_Kaggle_All.ipynb`**. File tự chứa mã nguồn đúng phiên bản đang chuẩn bị; không cần clone repo hay upload Dataset mã nguồn.

**Bản sửa setup cho phiên chạy của Đức:** `Lab22_Kaggle_All_Fixed.ipynb` đặt sẵn OpenAI `gpt-4.1-mini`,
HF repo `duckak12/lab22-dpo-vn-experimental`, và bật publish theo cấu hình bạn đã chọn.
Import thành notebook Kaggle mới, bật lại quyền Secrets; không giữ các cell vá cũ.
Bản này dùng `/kaggle/working/lab22-v2` và `/tmp/lab22-kaggle-venv-v2` để không đụng nguồn/môi trường cũ.
Cell A in `kaggle-v2-pip-bootstrap` để xác nhận đúng bản; cell D tự tạo venv và seed pip trước khi cài package.

## Trước khi chạy

1. Tạo/import notebook bằng file `.ipynb`, bật Internet, chọn GPU T4 ×2 nếu có.
2. Mặc định chỉ dùng GPU 0, tier T4, Qwen3-4B; hai GPU không cộng thành VRAM 32 GB cho cấu hình này.
3. Giữ `RUN_BONUS=True` để chạy tất cả notebook và β-sweep.
4. Muốn chấm chéo (+4): đặt `API_JUDGE_MODEL` ở cell cấu hình bằng model ID thật bạn có quyền gọi. Tạo secret theo provider: `GEMINI_API_KEY`, `OPENAI_API_KEY` hoặc `ANTHROPIC_API_KEY` và bật quyền cho notebook.
5. Muốn HF Hub (+3): đặt `HF_REPO="username/lab22-dpo-vn-experimental"`, thêm `HF_TOKEN` có quyền write, bật `PUBLISH_HF=True`. Hành động này upload **hai repo public**, adapter và SFT reference. Mô hình có dữ liệu SFT phi thương mại; đọc model card được tạo trước khi sử dụng ngoài học tập.
6. Không in key hoặc đặt key trực tiếp trong notebook. Phần chính và các bonus notebook không cần API key.

## Trong lúc chạy

- Chạy cell từ trên xuống. Cài đặt ban đầu tạo môi trường riêng trong `/tmp`; cần tải thư viện/mô hình nên có thể lâu.
- Mỗi stage tạo notebook có output ở `/kaggle/working/lab22/notebooks/`. Đọc output NB2 và tự nhận xét 5 cặp dữ liệu.
- Sau mỗi stage có `/kaggle/working/LAB22_RESULTS.zip`. Tải ngay sau phần chính, trước khi làm bonus dài.
- Nếu một core stage lỗi: dừng các bước phụ thuộc, gửi ZIP chứa notebook dở dang và log cho Codex.
- Nếu bonus lỗi: notebook giữ trạng thái FAILED và tiếp tục các bonus độc lập. FAILED/SKIPPED không được tính là đã hoàn thành.
- Với NB5: mở output notebook `05_merge_deploy_gguf.ipynb` hoặc xem cell smoke, chụp ảnh tên Q4_K_M + đầu ra; gửi để lưu `06-gguf-smoke.png`.
- Không đổi seed/MAX_LEN/split giữa các stage. Nếu phải đổi, lưu toàn bộ kết quả cũ và tạo thí nghiệm mới; không dùng marker skip của lượt cũ.

## Kết quả cần gửi lại

1. `LAB22_RESULTS.zip` — nguồn, các notebook đã chạy, hình, metrics, preference parquet, output và judge verdicts, trạng thái từng bonus. ZIP không có trọng số lớn.
2. Notebook Kaggle đã chạy và còn output, hoặc link Kaggle cho phép đọc notebook.
3. Ảnh chụp smoke GGUF thực tế (NB5).
4. Nếu có: link HF Hub; chúng cũng được ghi trong `data/eval/hub_publish.json`.

REFLECTION còn placeholder trước khi bạn gửi số liệu. Verify ở giai đoạn này sẽ báo UNEDITED; không sửa thành kết quả giả. Codex sẽ phân tích số liệu, điền REFLECTION, kiểm tra các điều kiện rubric, xử lý bonus thiếu/lỗi, rồi commit/push bài nộp khi đã đủ bằng chứng. Điểm cuối cùng do người chấm quyết định.

## Chuyển phiên Kaggle

Lưu version/output và kiểm tra có cả `lab22/models/sft-merged` + `lab22/adapters`. Mỗi session mới cần cài lại môi trường trong `/tmp`; dữ liệu checkpoint phải giữ ở nơi bền vững.

Tùy chọn `CREATE_CHECKPOINT_ZIP=True` tạo `LAB22_CHECKPOINT.zip` chứa weights và trạng thái. ZIP này lớn và cần chỗ trống để tạo thêm một bản sao; không gửi cho Codex nếu chỉ cần phân tích số liệu. Upload nó vào Dataset private, gắn vào notebook phiên sau và đặt `RESTORE_ARCHIVE` tới đường file đúng. Không dùng `LAB22_RESULTS.zip` làm checkpoint huấn luyện.

Tiếp tục bằng cùng notebook và cùng `/kaggle/working/lab22`. Các stage chỉ skip khi marker và artifacts khớp. Nếu đã sửa nguồn/cấu hình, chương trình từ chối tự ghi đè lượt thành công.

## Tình trạng kiểm chứng

Gói được kiểm tra cấu trúc notebook, source embedding, logic đóng gói và các test CPU tại máy chuẩn bị. Chưa chạy end-to-end trên GPU Kaggle; có thể cần sửa tương thích môi trường khi nhận log thật. Không có đảm bảo thời gian chạy hay điểm tối đa trước khi có kết quả.

LMS vẫn nhận **link GitHub public** theo rubric; không thay bằng link Kaggle. Sau phân tích, chỉ commit source/bằng chứng nhỏ, không commit weights/API key. BONUS-CHALLENGE không chấm điểm nằm ngoài phạm vi theo lựa chọn của bạn.
