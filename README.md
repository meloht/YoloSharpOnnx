![YoloSharpOnnx](https://socialify.git.ci/meloht/YoloSharpOnnx/image?description=1&forks=1&language=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fmeloht%2FYoloSharpOnnx%2Frefs%2Fheads%2Fmaster%2Fimgs%2Fyolo_logo.svg&name=1&owner=1&pulls=1&stargazers=1&theme=Light)
# YoloSharpOnnx
![YOLOv8-v26](https://img.shields.io/badge/YOLOv8--v26-supported-2ea44f) ![C#](https://img.shields.io/badge/language-C%23-blue.svg) ![.NET Version](https://img.shields.io/badge/dynamic/xml?url=https://raw.githubusercontent.com/meloht/YoloSharpOnnx/refs/heads/master/YoloSharpOnnx/YoloSharpOnnx.csproj&query=//TargetFrameworks&label=.NET) ![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blue.svg?logo=onnx&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg?logo=opencv&logoColor=white) ![GitHub license](https://img.shields.io/github/license/meloht/YoloSharpOnnx) ![Release](https://img.shields.io/github/v/release/meloht/YoloSharpOnnx.svg?label=Release) ![NuGet](https://img.shields.io/nuget/v/YoloSharpOnnx.svg?logo=nuget&logoColor=white)

🚀a high performance, memory reuse, production-ready C# YOLO inference library for object detection  base on OpenCV and ONNX Runtime.

# Features
 - **YOLO Task**  [Object Detection](https://docs.ultralytics.com/tasks/detect) 
 - **Execution Provider** CPU, CUDA / TensorRT, OpenVINO, CoreML, DirectML
 - **Batch processing images** Preprocess and Inference are executed asynchronously  with Producer/Consumer pattern
 - **High Performance Inference** Memory reuse, GPU Inference I/O Binding
 - **Image Processing** [OpenCvSharp4](https://github.com/shimat/opencvsharp)
 - **Inference Engine** [ONNX Runtime](https://github.com/microsoft/onnxruntime) is a cross-platform inference and training machine-learning accelerator.
 - **YOLO Versions** Includes support for: [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11) ,[YOLO26](https://docs.ultralytics.com/models/yolo26)


## Example Images:
<div align="center">
 
| Object Detection Result  |
|---------------|
| <img src="./ExampleImages/bus_detect.jpg" width="500" > |
| <img src="./ExampleImages/zidane_detect.jpg" width="500"> |

</div>

# Build Release x64


# Usage

### 1. Export model to ONNX format:

For convert the pre-trained PyTorch model to ONNX format, run the following Python code:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('path/to/best.pt')

# Export the model to ONNX format
model.export(format='onnx')
```

### 2. Load the ONNX model with C#:

Install Nuget package `YoloSharpOnnx`, `OnnxRuntime`, `OpenCvSharp4.runtime`

#### CPU Inference
```shell
dotnet add package YoloSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Microsoft.ML.OnnxRuntime
```

#### CoreML Inference
```shell
dotnet add package YoloSharpOnnx
dotnet add package OpenCvSharp4.runtime.osx.10.15-x64
dotnet add package Microsoft.ML.OnnxRuntime
```

#### CUDA/TensorRT Inference
```shell
dotnet add package YoloSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Microsoft.ML.OnnxRuntime.Gpu.Windows
```

#### DirectML Inference
```shell
dotnet add package YoloSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```

#### OpenVINO Inference
```shell
dotnet add package YoloSharpOnnx
dotnet add package OpenCvSharp4.runtime.win
dotnet add package Intel.ML.OnnxRuntime.OpenVino
```

#### Use the following C# code to load the model and run basic prediction:

```csharp

using Mat image = Cv2.ImRead("bus.jpg");
using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU("yolo11n.onnx"));

List<DetectionResult> res = yolo.RunDetect(image);

yolo.DrawDetections(image,res);
Cv2.ImWrite("bus_res.jpg", image);

string printString = YoloUtils.GetResult(res);
Console.WriteLine(printString)

```

#### YoloSharpOnnx performance testing api

```csharp

using Mat image = Cv2.ImRead("bus.jpg");
     
using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML("yolo11n.onnx",1));
var res = yolo.RunDetectWithTime(item.FullName);

Console.WriteLine($"{res.ToString()}, {res.SpeedResult.ToString()}");

```

#### Config 
```csharp
using Mat image = Cv2.ImRead("bus.jpg");
using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU("yolo11n.onnx"));
yolo.YoloConfiguration.IoU = 0.4f;
yolo.YoloConfiguration.Confidence = 0.3f;
yolo.YoloConfiguration.ResizeAlgorithm = InterpolationFlags.Linear;
yolo.YoloConfiguration.ImageExtsBatch = [".jpg", ".png"];
var res = yolo.RunDetect(image);
```

#### Batch processing images

```csharp
 private static void TestBatchInfer()
 {
     string modelPath = @"D:\code\model\best.onnx";
     string dir = @"D:\code\model\TestImages";

     DirectoryInfo directory = new DirectoryInfo(dir);
     var files = directory.GetFiles();

     System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
     _stopwatch.Start();
     using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, 1)))
     {
         yolo.BatchDetectItemCompleted += Yolo_BatchDetectItemCompleted;

         var list = yolo.RunBatchDetect(dir, 30);

     }
     _stopwatch.Stop();

     Console.WriteLine($"time:{_stopwatch.Elapsed}");
 }

 private static void Yolo_BatchDetectItemCompleted(object? sender, Models.BatchDetectionResultEventArgs e)
 {
     string ans = YoloUtils.GetResult(e.Results);
     Console.WriteLine(ans);
 }

```

# Roadmap

| Time  | Feature |
| ------------- | ------------- |
| 2026-10  | Yolo task Image Classification  |
| 2026-11  | Yolo task Instance Segmentation  |
| 2026-11  | Yolo task Pose Estimation  |
| 2026-12  | Yolo task OBB  |

# Model Licensing & Responsibility

* YoloSharpOnnx is licensed under the [MIT License](./LICENSE.txt) and provides an ONNX inference
engine for YOLO models exported using Ultralytics YOLO tooling.

* This project does **not** include, distribute, download, or bundle any
pretrained models.

* Users must supply their own ONNX models.

* YOLO ONNX models produced using Ultralytics tooling are typically licensed
under **AGPL-3.0** or a separate commercial license from Ultralytics.

* YoloSharpOnnx does **not** impose, modify, or transfer any license terms related
to user-supplied models.

* **Users are solely responsible** for ensuring that their use of any model
complies with the applicable license terms, including requirements related
to commercial use, distribution, or network deployment.
  
