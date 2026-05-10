using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.Detect.Models
{

    public class PreDetectChannelData : IGuidValue
    {
        public Guid Guid { get; init; }
        public PreDetectResultBatch PreResult { get; init; }

        public PreDetectChannelData(Guid guid, PreDetectResultBatch preResult)
        {
            Guid = guid;
            PreResult = preResult;
        }
    }
}
