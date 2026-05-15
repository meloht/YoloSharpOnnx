using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.OutputDecode
{
    internal class NmsDecode
    {
        private readonly List<Rect> _boxes;
        private readonly List<float> _scores;
        private readonly List<int> _classIds;
        private readonly List<int> _ids;
        private readonly int _numAnchors;

        private readonly int _numAnchors2;
        private readonly int _numAnchors3;
        private readonly int _numAnchors4;
        private readonly int _classCount;

        private readonly YoloConfig _yoloConfig;

        public NmsDecode(OnnxModel onnx, YoloConfig yoloConfig, List<Rect> boxes, List<float> scores, List<int> classIds, List<int> ids = null)
        {
            _numAnchors = (int)onnx.OutputShape0[2]; 
            _boxes = boxes;
            _scores = scores;
            _classIds = classIds;
            _ids = ids;
            _numAnchors2 = _numAnchors * 2;
            _numAnchors3 = _numAnchors * 3;
            _numAnchors4 = _numAnchors * 4;

            _classCount = onnx.Labels.Length;
            _yoloConfig = yoloConfig;
        }


        public int[] Decode(ReadOnlySpan<float> output0, PreDetectResult preResult)
        {
            for (int i = 0; i < _numAnchors; i++)
            {
                int classOffset = i + _numAnchors4;
                float maxScore = 0f;
                int classId = -1;

                for (var c = 0; c < _classCount; c++, classOffset += _numAnchors)
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
                    
                    _ids?.Add(i);
                }
            }

            // 非极大值抑制
            int[] indices = [];
            if (_boxes.Count > 0)
            {
                CvDnn.NMSBoxes(_boxes, _scores, _yoloConfig.Confidence, _yoloConfig.IoU, out indices);
            }
            return indices;
        }

    }
}
