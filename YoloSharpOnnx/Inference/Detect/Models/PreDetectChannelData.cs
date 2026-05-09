using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.Detect.Models
{


    public record PreDetectChannelData(PreDetectResultBatch PreResult, Guid Guid);
}
