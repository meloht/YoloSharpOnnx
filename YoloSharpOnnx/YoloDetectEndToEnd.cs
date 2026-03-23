using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloDetectEndToEnd : YoloDetectBase, IYoloDetect
    {

        public YoloDetectEndToEnd(InferenceSession session, SessionOptions options)
            : base(session, options)
        {

        }

        private void Preprocess(Mat image, float ratio, float[] data, int inputWidth, int inputHeight, InterpolationFlags resizeAlgorithm)
        {
            // 1. Preprocessing (Letterbox)
            int newWidth = (int)(image.Width * ratio);
            int newHeight = (int)(image.Height * ratio);

            using var resized = new Mat();
            Cv2.Resize(image, resized, new OpenCvSharp.Size(newWidth, newHeight), interpolation: resizeAlgorithm);

            using var canvas = new Mat(new OpenCvSharp.Size(inputWidth, inputHeight), MatType.CV_8UC3, _paddingColor);
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
                    ClassName = _labels[labelId].Name
                });
            }

            return detections;
        }

        public void Dispose()
        {
            _session.Dispose();
            _options.Dispose();
            _runOptions.Dispose();
        }

        public List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig)
        {
            float[] data = _inputBuffer;
            float ratio = Math.Min((float)_inputWidth / inputImage.Width, (float)_inputHeight / inputImage.Height);

            Preprocess(inputImage, ratio, data, _inputWidth, _inputHeight, yoloConfig.ResizeAlgorithm);
            // 3. 推理
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data, _inputShape);


            using var results = _session.Run(_runOptions, [_inputName], [inputOrtValue], _session.OutputNames);
            using var output0 = results[0];

            // 4. 后处理 (YOLO26 直接输出 [1, 300, 6])
            return PostProcess(output0, yoloConfig.Confidence, ratio);
        }
    }
}
