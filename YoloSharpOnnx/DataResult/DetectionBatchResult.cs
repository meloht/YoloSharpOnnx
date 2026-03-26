using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public record DetectionBatchResult(string ImagePath, List<DetectionResult> Results);
   
}
