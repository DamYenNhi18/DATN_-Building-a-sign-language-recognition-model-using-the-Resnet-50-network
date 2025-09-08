import splitfolders

input_folder =r"D:\DoAnTotNghiep\data_6"# Thay bằng đường dẫn thư mục dữ liệu của bạn
output_folder = r"D:\DoAnTotNghiep\split_data_6_2"# Thư mục lưu dữ liệu sau khi chia


# Chia dữ liệu thành 80% train, 20% test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.9,0, 0.1), group_prefix=None) # chia data thành 80% tập train và 20% test


print("Chia dữ liệu xong!")
