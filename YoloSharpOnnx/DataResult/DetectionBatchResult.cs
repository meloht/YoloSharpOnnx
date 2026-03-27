using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public class DetectionBatchResult
    {
        public string ImagePath { get; set; }

        public List<DetectionResult> Results { get; set; }

        public DetectionBatchResult(string imagePath, List<DetectionResult> results)
        {
            this.ImagePath = imagePath;
            this.Results = results;
        }

        public override string ToString()
        {
            return $"Image:{Path.GetFileName(ImagePath)} Result:{YoloUtils.GetResult(Results)}";
        }


    }



}
