# AIMI_hw2
## How to use
- 下載訓練資料和測試資料，並將檔案放置於 **lab2_dataset\train** 和 **lab2_dataset\test** 下
- 新增名為 **models** 的資料夾，並將 **EEGNet.py** 放置於該資料夾下
- 新增名為 **weights** 的資料夾
- 設定參數
  - num_epochs: epoch數 (預設: 150)
  - batch_size: batch的大小 (預設: 64)
  - lr: learning rate的大小 (預設: 0.01)
  - alpha: EEGNet中ELU function的alpha值 (預設: 1.0)
  - dropout: EEGNet中Dropout Layer的dropout機率 (預設: 0.25)
- 執行 `main.py`
