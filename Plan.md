Bối cảnh hiện tại: Đây là 1 project AI phát hiện đột quỵ phục vụ cho đồ án IOT.
+ Hiện tại AI đã có thể vẻ khung người khung xương để để có thể bắt chuyện động cũng như phát hiện và cảnh báo đột quỵ thông qua webcam và video có sẵn
+ Cũng đẫ có GUI trực quan 
+ Đã có thể kết nối với supabase để lưu trữ và truy xuất dữ liệu
Vấn đề:
+ Mặc dù đã có thể phát hiện người bị đột quỵ nhưng một số trường hợp thì bị phát hiện nhầm, người ngồi bình thường cũng bị báo động và chụp ảnh lại
+ Hiện tại việc đẩy ảnh lên supabase để lưu trữ nhưng hiện tại chỉ đang test AI nên có nhiều ảnh bị đẩy lên dư thừa cũng như cần phải xóa để sạch database
Yêu cầu:
+ Cải thiện AI để có thể phát hiện người bị đột quỵ chính xác hơn 
+ Trong GUI hiện thị thêm 1 chức năng đó là lấy dữ liệu trên database xuống để kiểm tra, cũng như xóa các dữ liệu không cần thiết trong database
