using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Classify.Models
{

    public class PreClsChannelData : IGuidValue<PreClsResultBatch>, IBatchPreChannelResult<PreClsResultBatch>
    {
        public Guid Guid { get; set; }
        public PreClsResultBatch PreResult { get; set; }

        public void Initialize(Guid guid, PreClsResultBatch preResult)
        {
            Guid = guid;
            PreResult = preResult;
        }
        public PreClsChannelData()
        {
        }

        public void Dispose()
        {
            PreResult = null;
        }
    }
}
