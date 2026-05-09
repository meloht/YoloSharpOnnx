using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect.Models
{
    public record PreDetectResultBatch(PreDetectResult PreResult, string ImagePath, ImageBatchData Data);
   
}
