using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public struct DetectionResult: IYoloPrediction<DetectionResult>
    {
        public Rect Box { get;  }
        public float Confidence { get;  }
        public int ClassId { get;  }
        public string ClassName { get;  }

        public DetectionResult(Rect box, float confidence, int classId, string className)
        {
            Box = box;
            Confidence = confidence;
            ClassId = classId;
            ClassName = className;
        }

        static string IYoloPrediction<DetectionResult>.Describe(DetectionResult[] predictions) => predictions.Summary();
    }
}
