251119_vigetting_label：标签图像生成文件，去除原始图像（raw_for_dl）的暗角，生成label_for_dl

label_for_dl：标签图像（jpg格式）

raw_for_dl：原始图像

251201_resunet.py：主要的模型训练文件。采用Adam优化器进行参数更新，损失函数采用L2损失

final_resunet_weights/resunet_restore_512_epoch_50.pth：训练50个Epoch后保存的最终模型

train_log.txt：完整的训练历史记录，记录了50轮训练中每一轮的Loss值变化。

restored_outputs_final_512：用最终模型处理原始图像的结果

evaluate.py：模型评估脚本，用标签图像和最终模型处理过的图像进行对比，计算PSNR/SSIM指标，SSIM值在0.96以上

PSNR_SSIM.bpm：X轴表示图像索引，Y轴表示PSNR，代表模型处理后的图片与标签图像之间的峰值信噪比，其值越高，修复效果越好。图中蓝线代表模型在每一张图像上的PSNR值。红线是平均PSNR，表示所有测试图像PSNR的平均水平

