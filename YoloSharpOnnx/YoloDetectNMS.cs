using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloDetectNMS : YoloDetectBase, IYoloDetect
    {

        private readonly int _boxNums;
        private readonly int _boxNums2;
        private readonly int _boxNums3;
        private readonly int _boxNums4;

        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();

        public YoloDetectNMS(InferenceSession session, SessionOptions options)
            : base(session, options)
        {

            _boxNums = (int)_outputShape[2];
            _boxNums2 = _boxNums * 2;
            _boxNums3 = _boxNums * 3;
            _boxNums4 = _boxNums * 4;
        }






        private List<DetectionResult> Postprocess(int imageHeight, int imageWidth, ReadOnlySpan<float> ortSpan, int padTop, int padLeft, float scale, YoloConfiguration yoloConfig)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();

            for (int i = 0; i < _boxNums; i++)
            {
                // Move forward to confidence value of first label
                var labelOffset = i + _boxNums4;

                float bestConfidence = 0f;
                int bestLabelIndex = -1;

                // Get confidence and label for current bounding box
                for (var l = 0; l < _labels.Length; l++, labelOffset += _boxNums)
                {
                    var boxConfidence = ortSpan[labelOffset];

                    if (boxConfidence > bestConfidence)
                    {
                        bestConfidence = boxConfidence;
                        bestLabelIndex = l;
                    }
                }

                // Stop early if confidence is low
                if (bestConfidence < yoloConfig.Confidence)
                    continue;

                float x = ortSpan[i] - padLeft;
                float y = ortSpan[i + _boxNums] - padTop;
                float w = ortSpan[i + _boxNums2];
                float h = ortSpan[i + _boxNums3];

                // Calculate the scaled coordinates of the bounding box
                int left = (int)((x - w / 2) / scale);
                int top = (int)((y - h / 2) / scale);
                int width = (int)(w / scale);
                int height = (int)(h / scale);

                // Ensure coordinates are within image bounds
                left = Math.Max(0, left);
                top = Math.Max(0, top);
                width = Math.Min(width, imageWidth - left);
                height = Math.Min(height, imageHeight - top);

                // Add the class ID, score, and box coordinates to the respective lists
                if (width > 0 && height > 0)
                {
                    _classIds.Add(bestLabelIndex);
                    _scores.Add(bestConfidence);
                    _boxes.Add(new Rect(left, top, width, height));
                }
            }

            // 非极大值抑制
            int[] indices = [];
            if (_boxes.Count > 0)
            {
                CvDnn.NMSBoxes(_boxes, _scores, yoloConfig.Confidence, yoloConfig.IoU, out indices);
            }
            List<DetectionResult> results = new List<DetectionResult>();
            // 绘制检测结果
            foreach (var idx in indices)
            {
                Rect box = _boxes[idx];
                float score = _scores[idx];
                int class_id = _classIds[idx];
                string lable = _labels[class_id].Name;

                DetectionResult detection = new DetectionResult();
                detection.Confidence = score;
                detection.ClassName = lable;
                detection.ClassId = class_id;
                detection.Box = box;
                results.Add(detection);

            }

            return results;
        }

        public void Dispose()
        {
            _session.Dispose();
            _options.Dispose();
            _runOptions.Dispose();
        }

        public List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig)
        {
            // 预处理图像
            var preRes = Preprocess(inputImage, _inputBuffer, yoloConfig.ResizeAlgorithm);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(preRes.OutData, _inputShape);

            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = Postprocess(inputImage.Height, inputImage.Width, output0.GetTensorDataAsSpan<float>(), preRes.PadY, preRes.PadX, preRes.Scale, yoloConfig);


            return result;
        }

        public YoloResult<DetectionResult> RunDetect(Mat inputImage, YoloConfiguration yoloConfig)
        {

            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
            var preRes = Preprocess(inputImage, _inputBuffer, yoloConfig.ResizeAlgorithm);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(preRes.OutData, _inputShape);
            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();


            // 后处理
            var res = Postprocess(inputImage.Height, inputImage.Width, output0.GetTensorDataAsSpan<float>(), preRes.PadY, preRes.PadX, preRes.Scale, yoloConfig);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }
    }
}
