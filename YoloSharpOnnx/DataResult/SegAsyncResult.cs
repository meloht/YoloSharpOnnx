using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference;

namespace YoloSharpOnnx.DataResult
{
    public class SegAsyncResult: IChannelAsyncResult<SegResult>
    {
        public Guid Guid { get; set; }

        public List<SegResult> Results { get; set; }

        /// <summary>
        /// DateTimeOffset.UtcNow.ToUnixTimeMilliseconds
        /// </summary>
        public long StartTimestamp { get; set; }

        public override string ToString()
        {
            return $"Guid:{Guid} Result:{Results.Summary()}";
        }

        public void Initialize(Guid guid, List<SegResult> results, long timestamp)
        {
            this.Guid = guid;
            this.Results = results;
            this.StartTimestamp = timestamp;
        }
    }
}
