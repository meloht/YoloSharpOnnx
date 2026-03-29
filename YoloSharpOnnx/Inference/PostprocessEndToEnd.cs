using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class PostprocessEndToEnd : IPostprocess
    {
        private readonly LabelModel[] _labels;
        public PostprocessEndToEnd(LabelModel[] labels)
        {
            _labels = labels;
        }
        public List<DetectionResult> PostProcess(OrtValue outputValue, PreResult preResult, YoloConfig yoloConfig)
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
                if (confidence < yoloConfig.Confidence) continue;

                // 3. 提取坐标并还原到原始图像尺寸
                // 注意：YOLOv26 默认输出通常是 [x1, y1, x2, y2]
                float x1 = (data[offset + 0] - preResult.PadX) / preResult.Scale;
                float y1 = (data[offset + 1] - preResult.PadY) / preResult.Scale;
                float x2 = (data[offset + 2] - preResult.PadX) / preResult.Scale;
                float y2 = (data[offset + 3] - preResult.PadY) / preResult.Scale;

                int labelId = (int)data[offset + 5];

                detections.Add(new DetectionResult()
                {
                    Box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1)),
                    Confidence = confidence,
                    ClassId = labelId,
                    ClassName = _labels[labelId].Name
                });
            }

            return detections;
        }
    }
}
