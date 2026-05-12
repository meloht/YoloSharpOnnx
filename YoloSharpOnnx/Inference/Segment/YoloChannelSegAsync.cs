using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class YoloChannelSegAsync: YoloChannelAsync<SegResult, PreDetectResultBatch, PreDetectChannelData>
    {
        public YoloChannelSegAsync(YoloConfig yoloConfig,
           IYoloProcessAsync<PreDetectResultBatch> yoloProcessAsync,
           IRunBatch<SegResult, PreDetectResultBatch> runBatch) : base(yoloConfig, yoloProcessAsync, runBatch)
        {

        }

        protected override PreDetectChannelData BuildPreChannelData(PreDetectResultBatch batchPreResult, Guid guid)
        {
            return new PreDetectChannelData(guid, batchPreResult);
        }

        protected override List<SegResult> RunBatch(PreDetectChannelData batchPreResult)
        {
            return _runBatch.RunBatch(batchPreResult.PreResult);
        }
    }
}
