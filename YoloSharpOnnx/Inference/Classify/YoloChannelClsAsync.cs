using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;


namespace YoloSharpOnnx.Inference.Classify
{
    public class YoloChannelClsAsync : YoloChannelAsync<ClsResult, PreClsResultBatch, PreClsChannelData>
    {
        public YoloChannelClsAsync(YoloConfig yoloConfig, IYoloProcessAsync<PreClsResultBatch> yoloProcessAsync, IRunBatch<ClsResult, PreClsResultBatch> runBatch)
            : base(yoloConfig, yoloProcessAsync, runBatch)
        {
        }

        protected override PreClsChannelData BuildPreChannelData(PreClsResultBatch batchPreResult, Guid guid)
        {
            return new PreClsChannelData(guid, batchPreResult);
        }

        protected override List<ClsResult> RunBatch(PreClsChannelData batchPreResult)
        {
            return _runBatch.RunBatch(batchPreResult.PreResult);
        }
    }
}
