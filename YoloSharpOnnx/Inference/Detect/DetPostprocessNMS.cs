using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;
using YoloSharpOnnx.Utils;

namespace YoloSharpOnnx.Inference.Detect
{
    internal class DetPostprocessNMS : IDetCorePostprocess<DetectionResult>
    {
        private readonly LabelModel[] _labels;
        private readonly NmsDecode _nmsDecode;
        private readonly List<Rect> _boxes = new List<Rect>();
        private readonly List<float> _scores = new List<float>();
        private readonly List<int> _classIds = new List<int>();
        private readonly Lazy<ObjectPoolArr<PostResultArray>> _postResultPool;

        public DetPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _nmsDecode = new NmsDecode(onnx, yoloConfig);
            _postResultPool = new Lazy<ObjectPoolArr<PostResultArray>>(() => new ObjectPoolArr<PostResultArray>(PostResultArray.CreateForDetect, yoloConfig.BatchPoolSize, YoloUtils.ClearList));
        }

        private List<DetectionResult> PostProcessBase(OrtValue outputValue, PreDetectResult preResult, List<Rect> boxes, List<float> scores, List<int> classIds)
        {

            var ortSpan = outputValue.GetTensorDataAsSpan<float>();//[1,84,8400]

            int[] indices = _nmsDecode.Decode(ortSpan, preResult, boxes, scores, classIds);

            List<DetectionResult> results = new List<DetectionResult>();
            // 绘制检测结果
            foreach (var idx in indices)
            {
                Rect box = boxes[idx];
                float score = scores[idx];
                int class_id = classIds[idx];
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

        public List<DetectionResult> PostProcessAsync(OrtValue outputValue, PreDetectResult preResult)
        {
            var arr = _postResultPool.Value.Rent();
            try
            {
                return PostProcessBase(outputValue, preResult, arr.Boxes, arr.Scores, arr.ClassIds);
            }
            finally
            {
                _postResultPool.Value.Return(arr);
            }
        }
        public List<DetectionResult> PostProcessSync(OrtValue outputValue, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            return PostProcessBase(outputValue, preResult, _boxes, _scores, _classIds);
        }

        public void Dispose()
        {
            if (_postResultPool.IsValueCreated)
            {
                _postResultPool.Value.Dispose();
            }
        }
    }
}
