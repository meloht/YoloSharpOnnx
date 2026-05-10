using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Classify.Models
{

    public class PreClsChannelData : IGuidValue
    {
        public Guid Guid { get; init; }
        public PreClsResultBatch PreResult { get; init; }

        public PreClsChannelData(Guid guid, PreClsResultBatch preResult)
        {
            Guid = guid;
            PreResult = preResult;
        }
    }
}
