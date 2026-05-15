using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class DetPostprocessNMS : IDetPostprocess
    {
        private readonly int _numAnchors;

        private readonly LabelModel[] _labels;

        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();
        private readonly YoloConfig _yoloConfig;

        private readonly NmsDecode _nmsDecode;

        public DetPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _numAnchors = (int)onnx.OutputShape0[2];
            _yoloConfig = yoloConfig;
            _nmsDecode = new NmsDecode(onnx, yoloConfig, _boxes, _scores, _classIds);
        }

        public List<DetectionResult> PostProcess(OrtValue outputValue, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            var ortSpan = outputValue.GetTensorDataAsSpan<float>();//[1,84,8400]

            int[] indices = _nmsDecode.Decode(ortSpan, preResult);

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
