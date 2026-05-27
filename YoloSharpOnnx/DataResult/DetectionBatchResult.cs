using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference.DetectCore;

namespace YoloSharpOnnx.DataResult
{
    public class DetectionBatchResult : IBatchResultInit<DetectionResult>, IBatchResultItems<DetectionResult>
    {
        public string ImagePath { get; set; }

        public List<DetectionResult> Results { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }


        public override string ToString()
        {
            return $"Image:{Path.GetFileName(ImagePath)} Result:{Results.Summary()}";
        }

        public void Initialize(string imagePath, List<DetectionResult> results, long timestamp)
        {
            this.ImagePath = imagePath;
            this.Results = results;
            this.StartTimestamp = timestamp;
        }
    }



}
