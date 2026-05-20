using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public class PoseBatchResult
    {
        public string ImagePath { get; set; }

        public List<PoseResult> Results { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }

        public PoseBatchResult(string imagePath, List<PoseResult> results, long timestamp)
        {
            this.ImagePath = imagePath;
            this.Results = results;
            this.StartTimestamp = timestamp;
        }

        public override string ToString()
        {
            return $"Image:{Path.GetFileName(ImagePath)} Result:{Results.Summary()}";
        }
    }
}
