using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class SegPostprocessNMS : SegPostprocessBase, ISegPostprocess
    {
        private readonly int _numAnchors;
        private readonly int _numAnchors2;
        private readonly int _numAnchors3;
        private readonly int _numAnchors4;

        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();
        private List<int> _ids = new List<int>();

        private readonly int _classAtts;


        public SegPostprocessNMS(int numAnchors, OnnxModel onnx, YoloConfig yoloConfig) : base(onnx, yoloConfig)
        {
            _numAnchors = numAnchors;
            _numAnchors2 = _numAnchors * 2;
            _numAnchors3 = _numAnchors * 3;
            _numAnchors4 = _numAnchors * 4;

            _classAtts = (int)onnx.OutputShape0[1] - _maskDim;//[1,116,8400] 116-32=84

        }
        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            _ids.Clear();

            var shape0 = outputValue0.GetTensorTypeAndShape().Shape; //  [1,116,8400]
            var shape1 = outputValue1.GetTensorTypeAndShape().Shape; //[1,32,160,160]

            var output0 = outputValue0.GetTensorDataAsSpan<float>();
            var output1 = outputValue1.GetTensorDataAsSpan<float>();

            for (int i = 0; i < _numAnchors; i++)
            {
                int classOffset = i + _numAnchors4;
                float maxScore = 0f;
                int classId = -1;

                for (var c = 0; c < _labels.Length; c++, classOffset += _numAnchors)
                {
                    float score = output0[classOffset];

                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = c;
                    }
                }
                if (maxScore < _yoloConfig.Confidence) continue;

                float cx = output0[i] - preResult.PadX;
                float cy = output0[_numAnchors + i] - preResult.PadY;
                float w = output0[_numAnchors2 + i];
                float h = output0[_numAnchors3 + i];

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
                    _classIds.Add(classId);
                    _scores.Add(maxScore);
                    _boxes.Add(new Rect(x, y, width, height));
                    _ids.Add(i);

                }
            }
            // 非极大值抑制
            int[] indices = [];
            if (_boxes.Count > 0)
            {
                CvDnn.NMSBoxes(_boxes, _scores, _yoloConfig.Confidence, _yoloConfig.IoU, out indices);
            }

            List<SegResult> results = new List<SegResult>();

            foreach (var idx in indices)
            {
                float[] maskCoeffs = new float[32];
                for (int m = 0; m < _maskDim; m++)
                {
                    maskCoeffs[m] = output0[(_classAtts + m) * _numAnchors + _ids[idx]];
                }

                SegResult result = BuildResult(_boxes[idx], _classIds[idx], _scores[idx], maskCoeffs, output1, preResult);

                results.Add(result);
            }

            return results;
        }
    }
}
