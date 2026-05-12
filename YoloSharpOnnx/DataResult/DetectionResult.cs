using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public class DetectionResult : IYoloResult, IYoloSummary<DetectionResult>
    {
        public Rect Box { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; }
        public string ClassName { get; set; }

        static string IYoloSummary<DetectionResult>.Describe(List<DetectionResult> predictions) => predictions.Summary();

        public override string ToString()
        {
            return $"{ClassName} {Confidence}";
        }
    }
}
