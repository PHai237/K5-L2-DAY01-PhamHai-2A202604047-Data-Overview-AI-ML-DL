# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
{
    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
  }

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record nói về cả bức ảnh vì đây là tác vụ phân loại ảnh, chỉ dự đoán ảnh thuộc lớp nào chứ không xác định từng vật thể. Kết quả dự đoán là lớp "cab" với score 0.510915

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp được định nghĩa sẵn theo taxonomy ImageNet-1K gồm 1.000 lớp và đi kèm checkpoint, chứ không phải do model tự nghĩ ra

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Vì class_id là để định danh các lớp bằng mã riêng trong taxonomy được chọn, và class_name để giúp ta biết lớp đó tên là gì, còn taxonomy_name để biết các lớp này thuộc hệ thống nhãn nào, tránh nhầm lẫn khi các taxonomy sử dụng cùng một class_id

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí để người gán nhãn chọn một lớp phù hợp, tránh việc mỗi người chọn theo cảm nhận riêng, dẫn đến kết quả gán nhãn không nhất quán

6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ là điểm dự đoán của model khi phân loại ảnh và vẫn có thể sai, trong khi ground truth là nhãn được xác nhận làm chuẩn dựa trên dữ liệu thực tế và quy trình gán nhãn

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
{
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  }

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái ở [385.33, 69.24], góc dưới-phải ở [498.92, 348.92], rộng 113.58px, cao 279.68px

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
threshold 0.20 có 17 prediction, threshold 0.6 có 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Người reviewer cần kiểm tra các box được bổ sung sau khi hạ ngưỡng. Trong quá trình đó, có thể phải sửa lại nhãn nếu sai lớp, xóa các box khoanh nhầm vật thể không tồn tại hoặc điều chỉnh kích thước và vị trí của box nếu chưa chính xác

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Khi gán nhãn, người gán nhãn cần vẽ bounding box ôm sát vật thể, không quá rộng hoặc quá hẹp, để xác định chính xác vị trí và kích thước của vật thể trong ảnh

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Chiếc nồi màu bạc trong ảnh `kitchen` bị bàn che khuất khoảng một nửa. Em sẽ kiểm tra guideline xem đã có quy tắc gán nhãn cho vật thể bị che khuất chưa. Nếu có thì áp dụng, còn nếu chưa có hoặc hướng dẫn chưa rõ thì emn sẽ hỏi Lab Coach để xác định cách gán nhãn phù hợp

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
{
  "instance_id": "kitchen-001",
  "class_name": "person",
  "score": 0.899318,
  "polygon_point_count": 348,
  "polygon_xy": [
    [446.0, 70.0],
    [445.0, 71.0],
    [444.0, 71.0],
    [443.0, 72.0],
    [442.0, 72.0]
  ]
}

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
So với bounding box chỉ bao quanh vật thể bằng hình chữ nhật, polygon có thể mô tả hình dạng vật thể sát hơn nhờ nhiều điểm tọa độ tạo thành đường viền

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` dùng để phân biệt từng vật thể riêng lẻ trong một ảnh. Nó không phải `class_id` vì `class_id` dùng để xác định vật thể thuộc lớp nào, cũng không phải `tracking_id` vì `tracking_id` dùng để theo dõi cùng một vật thể qua nhiều frame

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Khi vẽ mask, người gán nhãn cần bám sát đường viền của vật thể, không bao gồm vùng nền xung quanh để đảm bảo xác định chính xác hình dạng và phạm vi của vật thể

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Chiếc nồi màu bạc trong ảnh `kitchen` bị bàn che khuất khoảng 50%. Em sẽ kiểm tra guideline xem đã có quy tắc vẽ mask cho vật thể bị che khuất chưa. Nếu đã có thì làm theo, còn nếu chưa có hoặc hướng dẫn chưa rõ thì sẽ hỏi Lab Coach (escalation) để xác định cách vẽ mask phù hợp, tránh tự ý ước lượng phần bị che khuất

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn được xác nhận cho toàn bộ ảnh | Ảnh traffic có nhiều loại xe, nhưng model dự đoán cab (score 0.51) | Kiểm tra ảnh và chọn một nhãn phù hợp theo guideline đã thống nhất | Kiểm tra nhãn đúng guideline, yêu cầu sửa nếu sai |
| Phát hiện vật thể | Mỗi vật thể có nhãn lớp và bounding box | Người ở mép trái ảnh kitchen bị cắt một phần, khó xác định box | Gán nhãn, vẽ và chỉnh box ôm sát vật thể | Kiểm tra nhãn, vị trí box và vật thể bị bỏ sót |
| Instance segmentation | Mỗi vật thể có nhãn lớp và mask | Ảnh kitchen có chiếc nồi bạc bị bàn che khuất, khiến việc xác định biên mask gặp khó khăn | Gán nhãn và vẽ mask theo đường viền vật thể | Kiểm tra nhãn, biên mask và vùng tô sai |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Em chỉ sử dụng dữ liệu được phép trong phạm vi bài lab, không tự ý chia sẻ dữ liệu cá nhân hoặc dữ liệu nhạy cảm

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Em sẽ dừng xử lý và báo ngay cho Lab Coach để được hướng dẫn, không tự ý sử dụng dữ liệu ngoài phạm vi

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
