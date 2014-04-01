LCellsSegmentation
==================

效果图见http://ww2.sinaimg.cn/mw690/48e47b86tw1eezvc0nivsj208y0csjt5.jpg

自己改进的图像分割(image segmentation)/目标提取算法，基于能量优化GraphCuts算法，因此使用了别人写的maxflow算法代码(已ignore)，使用者可以直接使用OpenCV库里面的GraphCutsAPI。

需要输入文件：超向素预处理结果，记录在final_mark.txt中，为原图等尺寸标号集合(已ignore)。

用户输入：使用鼠标左键随意在目标上标记几笔，右键随意在背景上标记几笔。w运行，r重新勾画，ESC退出。需要OpenCV库。
