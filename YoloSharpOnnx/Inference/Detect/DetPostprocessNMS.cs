using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class DetPostprocessNMS : IDetPostprocess
    {
        private readonly int _numAnchors;
        private readonly int _numAnchors2;
        private readonly int _numAnchors3;
        private readonly int _numAnchors4;
        private readonly LabelModel[] _labels;


        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();
        private readonly YoloConfig _yoloConfig;

        public DetPostprocessNMS(int boxNum, LabelModel[] labels, YoloConfig yoloConfig)
        {
            _labels = labels;
            _numAnchors = boxNum;
            _numAnchors2 = _numAnchors * 2;
            _numAnchors3 = _numAnchors * 3;
            _numAnchors4 = _numAnchors * 4;
            _yoloConfig = yoloConfig;
        }

        public List<DetectionResult> PostProcess(OrtValue outputValue, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            var ortSpan = outputValue.GetTensorDataAsSpan<float>();//[1,84,8400]
            int classOffset = 0;
            for (int i = 0; i < _numAnchors; i++)
            {
                // Move forward to confidence value of first label
                classOffset = i + _numAnchors4;

                float bestConfidence = 0f;
                int bestLabelIndex = -1;

                // Get confidence and label for current bounding box
                for (var l = 0; l < _labels.Length; l++, classOffset += _numAnchors)
                {
                    var boxConfidence = ortSpan[classOffset];

                    if (boxConfidence > bestConfidence)
                    {
                        bestConfidence = boxConfidence;
                        bestLabelIndex = l;
                    }
                }

                // Stop early if confidence is low
                if (bestConfidence < _yoloConfig.Confidence)
                    continue;

                float cx = ortSpan[i] - preResult.PadX;
                float cy = ortSpan[i + _numAnchors] - preResult.PadY;
                float w = ortSpan[i + _numAnchors2];
                float h = ortSpan[i + _numAnchors3];

                // Calculate the scaled coordinates of the bounding box
                int x = (int)((cx - w / 2f) / preResult.Scale);
                int y = (int)((cy - h / 2f) / preResult.Scale);
                int width = (int)(w / preResult.Scale);
                int height = (int)(h / preResult.Scale);

                // Ensure coordinates are within image bounds
                x = Math.Max(0, x);
                y = Math.Max(0, y);
                width = Math.Min(width, preResult.ImageWidth - x);
                height = Math.Min(height, preResult.ImageHeight - y);

                // Add the class ID, score, and box coordinates to the respective lists
                if (width > 0 && height > 0)
                {
                    _classIds.Add(bestLabelIndex);
                    _scores.Add(bestConfidence);
                    _boxes.Add(new Rect(x, y, width, height));
                }
            }

            // 非极大值抑制
            int[] indices = [];
            if (_boxes.Count > 0)
            {
                CvDnn.NMSBoxes(_boxes, _scores, _yoloConfig.Confidence, _yoloConfig.IoU, out indices);
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
