using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.DetectCore;

namespace YoloSharpOnnx.DataResult
{
    public class ObbBatchResult: IBatchResultInit<ObbResult>, IBatchResultItems<ObbResult>
    {
        public string ImagePath { get; set; }

        public List<ObbResult> Results { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }

        public override string ToString()
        {
            return $"Image:{Path.GetFileName(ImagePath)} Result:{Results.Summary()}";
        }

        public void Initialize(string imagePath, List<ObbResult> results, long timestamp)
        {
            this.ImagePath = imagePath;
            this.Results = results;
            this.StartTimestamp = timestamp;
        }
    }
}
