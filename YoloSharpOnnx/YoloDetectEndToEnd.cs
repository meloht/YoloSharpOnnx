using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class YoloDetectEndToEnd: YoloDetectBase
    {
        private void Preprocess(Mat image, float ratio, float[] data, int inputWidth, int inputHeight)
        {
            // 1. Preprocessing (Letterbox)
            int newWidth = (int)(image.Width * ratio);
            int newHeight = (int)(image.Height * ratio);

            using var resized = new Mat();
            Cv2.Resize(image, resized, new OpenCvSharp.Size(newWidth, newHeight));

            using var canvas = new Mat(new OpenCvSharp.Size(inputWidth, inputHeight), MatType.CV_8UC3, new Scalar(114, 114, 114));
            resized.CopyTo(new Mat(canvas, new Rect(0, 0, newWidth, newHeight)));

            // 2. 归一化并转换为 Tensor (HWC -> CHW)
            GetChwArr(canvas, data);
        }


        public List<DetectionResult> PostProcess(OrtValue outputValue, float threshold, float ratio)
        {
            var detections = new List<DetectionResult>();

            // 1. 获取第一个输出张量
            var shape = outputValue.GetTensorTypeAndShape().Shape; // 例如 [1, 300, 6]

            int rowCount = (int)shape[1]; // 300
            int colCount = (int)shape[2]; // 6

            // 2. 使用 Span 直接访问内存，避免产生垃圾回收
            ReadOnlySpan<float> data = outputValue.GetTensorDataAsSpan<float>();

            for (int i = 0; i < rowCount; i++)
            {
                // 计算当前行的偏移量
                int offset = i * colCount;

                float confidence = data[offset + 4];

                // 过滤低置信度结果
                if (confidence < threshold) continue;

                // 3. 提取坐标并还原到原始图像尺寸
                // 注意：YOLOv26 默认输出通常是 [x1, y1, x2, y2]
                float x1 = data[offset + 0] / ratio;
                float y1 = data[offset + 1] / ratio;
                float x2 = data[offset + 2] / ratio;
                float y2 = data[offset + 3] / ratio;

                int labelId = (int)data[offset + 5];

                detections.Add(new DetectionResult
                {
                    Box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1)),
                    Confidence = confidence,
                    ClassId = labelId,
                    ClassName = Labels[labelId].Name
                });
            }

            return detections;
        }

    }
}
