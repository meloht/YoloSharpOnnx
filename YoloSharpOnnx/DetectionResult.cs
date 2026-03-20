using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public struct DetectionResult
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
    }
}
