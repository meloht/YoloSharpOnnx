using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Segment.Models
{
    internal class InferModel : IDisposable
    {
        public void Initialize(OrtValue output0, OrtValue output1, string imagePath, long startTime, PreDetectResult preDetectResult)
        {
            Output0 = output0;
            Output1 = output1;
            ImagePath = imagePath;
            StartTime = startTime;
            PreDetectResult = preDetectResult;
        }
        public InferModel()
        {
        }
        public OrtValue Output0 { get; set; }
        public OrtValue Output1 { get; set; }
        public string ImagePath { get; set; }
        public long StartTime { get; set; }
        public PreDetectResult PreDetectResult { get; set; }

        public void Dispose()
        {
            Output0 = null;
            Output1 = null;
            ImagePath = null;
            StartTime = 0;
        }
    }
}
