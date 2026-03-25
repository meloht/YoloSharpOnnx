using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx.Models
{
    public class BatchDetectionResultEventArgs : EventArgs
    {
        public string ImagePath { get; set; }

        public List<DetectionResult> Results { get; set; }

        public BatchDetectionResultEventArgs(string path, List<DetectionResult> results)
        {
            ImagePath = path;
            Results = results;
        }
    }
}
