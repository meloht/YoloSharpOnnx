using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class PostprocessNMS : IPostprocess
    {
        private readonly int _boxNums;
        private readonly int _boxNums2;
        private readonly int _boxNums3;
        private readonly int _boxNums4;
        private readonly LabelModel[] _labels;


        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();

        public PostprocessNMS(int boxNum, LabelModel[] labels)
        {
            _labels = labels;
            _boxNums = boxNum;
            _boxNums2 = _boxNums * 2;
            _boxNums3 = _boxNums * 3;
            _boxNums4 = _boxNums * 4;
        }

        public List<DetectionResult> PostProcess(OrtValue outputValue, PreResult preResult, YoloConfiguration yoloConfig)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            var ortSpan = outputValue.GetTensorDataAsSpan<float>();

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

                float x = ortSpan[i] - preResult.PadX;
                float y = ortSpan[i + _boxNums] - preResult.PadY;
                float w = ortSpan[i + _boxNums2];
                float h = ortSpan[i + _boxNums3];

                // Calculate the scaled coordinates of the bounding box
                int left = (int)((x - w / 2) / preResult.Scale);
                int top = (int)((y - h / 2) / preResult.Scale);
                int width = (int)(w / preResult.Scale);
                int height = (int)(h / preResult.Scale);

                // Ensure coordinates are within image bounds
                left = Math.Max(0, left);
                top = Math.Max(0, top);
                width = Math.Min(width, preResult.ImageWidth - left);
                height = Math.Min(height, preResult.ImageHeight - top);

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
    }
}
