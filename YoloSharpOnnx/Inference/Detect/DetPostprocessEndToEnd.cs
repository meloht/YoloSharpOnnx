using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    internal class DetPostprocessEndToEnd : IDetPostprocess
    {
        private readonly LabelModel[] _labels;
        private readonly YoloConfig _yoloConfig;
        private readonly int _rowCount;
        private readonly int _colCount;
        public DetPostprocessEndToEnd(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _yoloConfig = yoloConfig;
            _rowCount = (int)onnx.OutputShape0[1];//[1,300,6]
            _colCount = (int)onnx.OutputShape0[2];//[1,300,6]
        }
        private List<DetectionResult> PostProcess(OrtValue outputValue, PreDetectResult preResult)
        {
            var detections = new List<DetectionResult>();

            // 2. 使用 Span 直接访问内存，避免产生垃圾回收
            ReadOnlySpan<float> data = outputValue.GetTensorDataAsSpan<float>();

            for (int i = 0; i < _rowCount; i++)
            {
                // 计算当前行的偏移量
                int offset = i * _colCount;

                float confidence = data[offset + 4];

                // 过滤低置信度结果
                if (confidence < _yoloConfig.Confidence) continue;

                // 3. 提取坐标并还原到原始图像尺寸

                // 读取6个基础属性
                Rect box = EndToEndDecode.Decode(data, offset, preResult);

                int labelId = (int)data[offset + 5];

                detections.Add(new DetectionResult()
                {
                    Box = box,
                    Confidence = confidence,
                    ClassId = labelId,
                    ClassName = _labels[labelId].Name
                });
            }

            return detections;
        }

        public List<DetectionResult> PostProcessAsync(OrtValue outputValue, PreDetectResult preResult)
        {
            return PostProcess(outputValue, preResult);
        }

        public List<DetectionResult> PostProcessSync(OrtValue outputValue, PreDetectResult preResult)
        {
            return PostProcess(outputValue, preResult);
        }

        public void Dispose()
        {
           
        }
    }
}
