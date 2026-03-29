using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public class DetectionBatchResult
    {
        public string ImagePath { get; set; }

        public List<DetectionResult> Results { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }

        public DetectionBatchResult(string imagePath, List<DetectionResult> results, long timestamp)
        {
            this.ImagePath = imagePath;
            this.Results = results;
            this.StartTimestamp = timestamp;
        }

        public override string ToString()
        {
            return $"Image:{Path.GetFileName(ImagePath)} Result:{YoloUtils.GetResult(Results)}";
        }


    }



}
