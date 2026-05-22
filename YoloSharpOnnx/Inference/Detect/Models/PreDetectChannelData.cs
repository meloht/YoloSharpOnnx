using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.Detect.Models
{

    internal class PreDetectChannelData : IGuidValue<PreDetectResultBatch>, IBatchPreChannelResult<PreDetectResultBatch>
    {
        public Guid Guid { get; set; }
        public PreDetectResultBatch PreResult { get; set; }

        public void Initialize(Guid guid, PreDetectResultBatch preResult)
        {
            Guid = guid;
            PreResult = preResult;
        }
        public PreDetectChannelData()
        {
        }
        public void Dispose()
        {
            PreResult = null;
        }
    }
}
