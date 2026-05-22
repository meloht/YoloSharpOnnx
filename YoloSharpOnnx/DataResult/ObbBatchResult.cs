using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.DataResult
{
    public class ObbBatchResult
    {
        public string ImagePath { get; set; }

        public List<ObbResult> Results { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }

        public ObbBatchResult(string imagePath, List<ObbResult> results, long timestamp)
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
