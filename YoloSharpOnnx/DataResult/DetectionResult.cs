using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public struct DetectionResult : IYoloPrediction<DetectionResult>
    {
        public Rect Box { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; }
        public string ClassName { get; set; }

        public DetectionResult(Rect box, float confidence, int classId, string className)
        {
            Box = box;
            Confidence = confidence;
            ClassId = classId;
            ClassName = className;
        }

        static string IYoloPrediction<DetectionResult>.Describe(List<DetectionResult> predictions) => predictions.Summary();

        public override string ToString()
        {
            return $"{ClassName} {Confidence}";
        }
    }
}
