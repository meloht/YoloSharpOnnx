using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference.Classify.Models
{

    public record PreClsChannelData(PreClsResultBatch PreResult, Guid Guid);
}
