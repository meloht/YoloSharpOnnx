using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloChannelDetectAsync : YoloChannelAsync<DetectionResult, PreDetectResultBatch, PreDetectChannelData>
    {
        public YoloChannelDetectAsync(YoloConfig yoloConfig,
            IYoloProcessAsync<PreDetectResultBatch> yoloProcessAsync,
            IRunBatch<DetectionResult, PreDetectResultBatch> runBatch) : base(yoloConfig, yoloProcessAsync, runBatch)
        {

        }

        protected override PreDetectChannelData BuildPreChannelData(PreDetectResultBatch batchPreResult, Guid guid)
        {
            return new PreDetectChannelData(guid, batchPreResult);
        }

        protected override List<DetectionResult> RunBatch(PreDetectChannelData batchPreResult)
        {
            return _runBatch.RunBatch(batchPreResult.PreResult);
        }
    }
}
