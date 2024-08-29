Install
1. Download the compressed file to your local machine.  
2. After extracting, type `cmd` in the address bar and press Enter to open the command prompt.  
3. If necessary, you can create a conda environment (optional).  
4. Install the required dependencies.  
   4.1 pip install ultralytics
   4.2 pip install pyqt6
6. Run the program by entering `python TrModel_beta8.py` in the command line.

Usage Instructions
0. The first time you open the program, it may continue to install some dependencies (I also forgot which dependencies they are).  
1. Select the folder path for the input video and the output folder (the output will be in AVI format and a CSV file).  
2. Choose the detection mode:  
   2.1 **Object Distribution Mode**: Modified based on Ultralytics' heatmap module, it can export the number of detected objects in each frame, thereby indicating the frequency of changes in the number of objects in that space. (This can be used for counting foot traffic in shopping malls and detecting the distribution of people.)  
   2.2 **Object Counting Mode**: Modified based on Ultralytics' object counting module, it can export the number of objects that have crossed the line up to the current frame. (This can be used to detect the influx and outflux of people in spaces such as shopping malls and train stations.)  
   2.3 **Object Speed Estimation Mode**: Modified based on Ultralytics' speed estimation module, it can export the speed of objects in the current frame. (This can record the walking speed of pedestrians over different time periods.) However, this module has not been tested yet, so please refer to it with caution.  
3. Choose the detection line mode (this option can be ignored in object distribution mode):  
   3.1 Select vertical or horizontal.  
   3.2 Adjust the position of the detection line by dragging the slider (if choosing vertical detection, try to position the detection line towards the wider area).  
4. Select the detection model (the default is yolov8n.pt, but you can modify it if needed).  
5. Choose the detection accuracy (adjust the detection accuracy by controlling the frame interval; 1 means detecting every frame, while 15 means detecting every 15 frames, which is intended to improve video detection efficiency, but currently, the results are not very satisfactory).  
6. Start running.
**Note:**  
1. The detection time is approximately twice the duration of the video.  
2. Videos in the same folder must have the same resolution.  
3. Ensure that all dependencies are installed correctly, especially the installation of PyTorch.
   
安装方法
1、下载压缩文件到本地
2、解压后，在地址栏位置输入cmd，回车，打开命令提示符
3、如有需要，可以自行创建conda环境（可选）
4、安装所需依赖
   4.1 pip install ultralytics
   4.2 pip install pyqt6
5、在命令行输入python TrModel_beta8.py运行程序

使用方法：
0、首次打开程序后，可能会继续安装一些依赖（我也忘记是哪些依赖了）
1、选择输入视频的文件夹地址和输出文件夹的地址（输出avi格式的视频，以及csv表格）
2、选择检测模式
2.1物体分布模式：在ultralytics的heatmap模块基础上进行修改，可以导出每一帧检测到的物体数量，从而得知该空间内物体数量变化的频率。（可用于商场内的人流数量统计以及人流分布检测）
2.2物体计数模式：在ultralytics的objectcounting模块基础上进行修改，可以导出记录到当前帧截至的、过线物体数量的多寡。（可用于检测诸如商场、火车站等空间，人流出入的数量增长曲线）
2.3物体测速模式：在ultralytics的speetestimat模块基础上进行修改，可以导出当前帧下，物体的速度。（可以记录不同时间段的，行人步行速度）但是该模块还未进行测试，请谨慎参考
3、选择检测线的模式（物体分布模式可以忽略该选项）
3.1选择纵向还是横向
3.2通过拖动滑块，选择检测线的位置（如果选择纵向检测的话，尽量把检测线往大的调）
4、选择检测模型（默认yolov8n.pt即可，如有需求可以自行修改）
5、选择检测精度（通过控制检测的间隔帧，来调整检测精度，1表示每帧检测一次，15表示每15帧检测一次，用于提高视频检测效率，但是目前看下来效果不佳）
6、开始运行

注意：
1、检测时间约为视频时常的两倍
2、同个文件夹内的视频分辨率必须相同
3、注意安装好依赖，尤其是pytorch的安装
